SCENARIOS = {
    "exploding_gradients": {
        "failure_mode": "exploding_gradients",
        "difficulty": "easy",
        "config": {
            "learning_rate": 10.0,
            "optimizer": "adam",
            "batch_size": 32,
            "weight_decay": 0.0,
            "architecture": "ResNet18",
            "dataset": "CIFAR-10"
        },
        "logs": [
            {"epoch": 1, "train_loss": 2.31, "val_loss": 2.35, "lr": 10.0},
            {"epoch": 2, "train_loss": 847.2, "val_loss": 912.4, "lr": 10.0},
            {"epoch": 3, "train_loss": float("nan"), "val_loss": float("nan")},
        ],
        "gradient_norms": None,   # not visible until agent requests it
        "correct_diagnosis": "exploding_gradients",
        "correct_fix": "reduce learning_rate to 0.001",
        "requires_fix": False,   # set True on hard scenarios where fix must be graded
    }
}
# TODO : Add more scenarios