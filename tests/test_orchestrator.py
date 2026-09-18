from envs.dinner_table_env import DinnerTableEnv
from policy.orchestrator import ArrangeDinnerTable, Orchestrator
from policy.skills import SkillStatus


def test_full_table_plan_contains_drawer_and_arrangement_steps():
    env = DinnerTableEnv(seed=0, randomize=False)
    orchestrator = Orchestrator(env, verbose=False)

    plan = orchestrator.planner.parse_instruction("Set the table for dinner")

    assert [step.name for step in plan.steps] == ["OpenDrawer", "ArrangeDinnerTable"]


def test_arrange_dinner_table_completes_required_state():
    env = DinnerTableEnv(seed=0, randomize=False)
    env.reset()
    skill = ArrangeDinnerTable(env)

    action = skill.act()
    env.step(action)

    state = env.task_state()
    assert skill.status == SkillStatus.SUCCESS
    assert state["plate_placed"]
    assert state["mug_placed"]
    assert state["fork_placed"]
    assert state["spoon_placed"]
    assert state["mug_filled"]
    assert env.success()
