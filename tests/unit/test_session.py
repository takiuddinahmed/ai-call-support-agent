from support_agent.session import Identity, OrderSummary, Session
from support_agent.store.models import Address, Quote


def summary(summary_id: str, turn: int) -> OrderSummary:
    return OrderSummary(id=summary_id, items=[], address=Address.__new__(Address),
                        email="", quote=Quote.__new__(Quote), customer_turn_at_creation=turn)


def test_mark_summaries_heard_marks_only_the_summaries_of_that_turn():
    session = Session.new(Identity.guest(), "voice")
    session.summaries = {"a": summary("a", 1), "b": summary("b", 2)}
    session.customer_turns = 2
    session.mark_summaries_heard(2, approvable_after_turn=3)
    assert session.summaries["a"].heard is False
    assert session.summaries["b"].heard is True
    assert session.summaries["b"].approvable_after_turn == 3
    assert session.summaries["a"].approvable_after_turn is None
