from typing import List

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from atrag.db.models import Evaluation, EvaluationItem, EvaluationItemStatus, EvaluationStatus, Question
from atrag.db.repositories.base import AsyncRepositoryProtocol
from atrag.exceptions import EvaluationQuotaExceededException
from atrag.utils.utils import utc_now


class AsyncEvaluationRepositoryMixin(AsyncRepositoryProtocol):
    async def create_evaluation_with_limits(
        self,
        evaluation: Evaluation,
        questions: List[Question],
        *,
        max_questions: int,
        max_active_evaluations: int,
        max_daily_items: int,
    ) -> Evaluation:
        """Create an evaluation while atomically enforcing user-level limits."""

        async def _operation(session: AsyncSession):
            bind = session.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:lock_name))"),
                    {"lock_name": f"atrag-evaluation-quota:{evaluation.user_id}"},
                )

            question_count = len(questions)
            if question_count > max_questions:
                raise EvaluationQuotaExceededException("max_questions_per_evaluation", max_questions, question_count)

            active_stmt = select(func.count(Evaluation.id)).where(
                Evaluation.user_id == evaluation.user_id,
                Evaluation.status.in_([EvaluationStatus.PENDING, EvaluationStatus.RUNNING]),
                Evaluation.gmt_deleted.is_(None),
            )
            active_count = (await session.execute(active_stmt)).scalar_one()
            if active_count >= max_active_evaluations:
                raise EvaluationQuotaExceededException(
                    "max_running_evaluations", max_active_evaluations, active_count
                )

            now = utc_now()
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            daily_stmt = select(func.coalesce(func.sum(Evaluation.total_questions), 0)).where(
                Evaluation.user_id == evaluation.user_id,
                Evaluation.gmt_created >= day_start,
                Evaluation.gmt_deleted.is_(None),
            )
            daily_count = int((await session.execute(daily_stmt)).scalar_one())
            if daily_count + question_count > max_daily_items:
                raise EvaluationQuotaExceededException("max_daily_evaluation_items", max_daily_items, daily_count)

            session.add(evaluation)
            await session.flush()
            await session.refresh(evaluation)
            session.add_all(
                [
                    EvaluationItem(
                        evaluation_id=evaluation.id,
                        question_id=question.id,
                        question_text=question.question_text,
                        ground_truth=question.ground_truth,
                        status=EvaluationItemStatus.PENDING,
                    )
                    for question in questions
                ]
            )
            await session.flush()
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_execution_context(
        self, evaluation_id: str, item_id: str
    ) -> tuple[Evaluation, EvaluationItem] | None:
        """Load an evaluation and one of its items for an authenticated worker."""

        async def _query(session: AsyncSession):
            stmt = (
                select(Evaluation, EvaluationItem)
                .join(EvaluationItem, EvaluationItem.evaluation_id == Evaluation.id)
                .where(
                    Evaluation.id == evaluation_id,
                    EvaluationItem.id == item_id,
                    Evaluation.gmt_deleted.is_(None),
                )
            )
            row = (await session.execute(stmt)).first()
            return (row[0], row[1]) if row else None

        return await self._execute_query(_query)

    async def create_evaluation(self, evaluation: Evaluation, questions: List[Question]) -> Evaluation:
        """Creates a new evaluation."""

        async def _operation(session: AsyncSession):
            # Insert evaluation into db and retrieve the id
            session.add(evaluation)
            await session.flush()
            await session.refresh(evaluation)

            for question in questions:
                eval_item = EvaluationItem(
                    evaluation_id=evaluation.id,
                    question_id=question.id,
                    question_text=question.question_text,
                    ground_truth=question.ground_truth,
                    status=EvaluationItemStatus.PENDING,
                )
                session.add(eval_item)

            await session.flush()
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def retry_evaluation(self, eval_id: str, user_id: str, scope: str) -> Evaluation | None:
        """Retries items in an evaluation based on the scope."""

        async def _operation(session: AsyncSession):
            # 1. Get the evaluation and lock it for update
            stmt = (
                select(Evaluation)
                .where(Evaluation.id == eval_id, Evaluation.user_id == user_id, Evaluation.gmt_deleted.is_(None))
                .with_for_update()
            )
            evaluation = (await session.execute(stmt)).scalar_one_or_none()

            if not evaluation:
                return None

            # 2. Reset status of items based on scope
            if scope == "all":
                items_stmt = (
                    update(EvaluationItem)
                    .where(EvaluationItem.evaluation_id == eval_id)
                    .values(
                        status=EvaluationItemStatus.PENDING,
                        rag_answer=None,
                        rag_answer_details=None,
                        llm_judge_score=0,
                        llm_judge_reasoning="",
                    )
                )
            else:  # scope == "failed"
                items_stmt = (
                    update(EvaluationItem)
                    .where(
                        EvaluationItem.evaluation_id == eval_id, EvaluationItem.status == EvaluationItemStatus.FAILED
                    )
                    .values(
                        status=EvaluationItemStatus.PENDING,
                        rag_answer=None,
                        rag_answer_details=None,
                        llm_judge_score=0,
                        llm_judge_reasoning="",
                    )
                )
            result = await session.execute(items_stmt)

            # 3. Reset evaluation status to RUNNING to allow re-processing.
            evaluation.status = EvaluationStatus.RUNNING
            if scope == "all":
                evaluation.completed_questions = 0
                evaluation.average_score = 0
            else:
                evaluation.completed_questions = evaluation.completed_questions - result.rowcount
            await session.flush()
            await session.refresh(evaluation)

            return evaluation

        return await self.execute_with_transaction(_operation)

    async def update_evaluation_status(
        self, eval_id: str, user_id: str, new_status: EvaluationStatus, current_statuses: List[EvaluationStatus] = None
    ) -> Evaluation | None:
        """Updates the status of an evaluation, optionally checking its current status."""

        async def _operation(session: AsyncSession):
            # 1. Fetch the evaluation to ensure it exists and belongs to the user.
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            if current_statuses:
                stmt = stmt.where(Evaluation.status.in_(current_statuses))

            evaluation = (await session.execute(stmt)).scalar_one_or_none()

            if not evaluation:
                return None

            # 2. Update the status.
            evaluation.status = new_status
            await session.flush()
            await session.refresh(evaluation)
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_items_by_eval_id(self, eval_id: str) -> list[EvaluationItem]:
        """Gets all evaluation items for a given evaluation."""

        async def _query(session: AsyncSession):
            stmt = (
                select(EvaluationItem)
                .where(EvaluationItem.evaluation_id == eval_id)
                .order_by(EvaluationItem.gmt_created.asc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_evaluation_by_id(self, eval_id: str, user_id: str) -> Evaluation | None:
        """Gets an evaluation by its ID."""

        async def _query(session: AsyncSession):
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def list_evaluations_by_user(
        self, user_id: str, collection_id: str | None, page: int, page_size: int
    ) -> tuple[list[Evaluation], int]:
        """Lists all evaluations for a user."""

        async def _query(session: AsyncSession):
            where_conditions = [
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            ]
            if collection_id:
                where_conditions.append(Evaluation.collection_id == collection_id)

            stmt = (
                select(Evaluation)
                .where(*where_conditions)
                .offset((page - 1) * page_size)
                .limit(page_size)
                .order_by(Evaluation.gmt_created.desc())
            )
            result = await session.execute(stmt)
            items = result.scalars().all()

            count_stmt = select(func.count(Evaluation.id)).where(*where_conditions)
            total = await session.scalar(count_stmt)

            return items, total

        return await self._execute_query(_query)

    async def delete_evaluation_by_id(self, eval_id: str, user_id: str) -> bool:
        """Deletes an evaluation by its ID."""

        async def _operation(session: AsyncSession):
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            db_evaluation = result.scalars().first()

            if not db_evaluation:
                return False

            db_evaluation.gmt_deleted = utc_now()
            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)
