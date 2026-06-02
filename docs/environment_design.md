# Environment Design

The initial environment uses abstract point-mass agents in a bounded 2D world. Defenders are controlled by continuous velocity actions. Intruders follow a simple scripted motion toward a protected center.

Termination occurs when all intruders are captured, any intruder reaches the protected zone, or the episode reaches the maximum step limit.

The design is intentionally independent of hardware, communication stacks, and real-world command interfaces.
