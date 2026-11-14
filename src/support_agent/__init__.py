"""Local AI customer support agent for the demo store."""


def main() -> None:
    from support_agent.server.app import run

    run()
