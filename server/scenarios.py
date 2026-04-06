SCENARIOS: dict[str, dict] = {

    # ─── EASY — identifiable from logs alone ─────────────────────────────────
    # Each scenario exercises a different keyword path in grade():
    #   exploding_gradients   → exact: "exploding" | category: "nan"/"diverge"
    #   learning_rate_too_high→ exact: "lr too high" | category: "oscillat"/"unstable"
    #   overfitting           → exact: "overfit"    | category: "val loss"/"memoriz"
    #   underfitting          → exact: "underfit"   | category: "plateau"/"high bias"

    "exploding_gradients": {
        "failure_mode": "exploding_gradients",
        "difficulty": "easy",
        "required_sources": ["logs"],
        "config": {
            "learning_rate": 10.0, "optimizer": "adam",
            "batch_size": 32, "weight_decay": 0.0,
            "architecture": "ResNet18", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1, "train_loss": 2.31, "val_loss": 2.35, "lr": 10.0},
            {"epoch": 2, "train_loss": 847.2, "val_loss": 912.4, "lr": 10.0},
            {"epoch": 3, "train_loss": float("nan"), "val_loss": float("nan")},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "exploding_gradients",
        "correct_fix": "reduce learning_rate to 0.001",
        "requires_fix": False,
    },

    "learning_rate_too_high": {
        "failure_mode": "learning_rate_too_high",
        "difficulty": "easy",
        "required_sources": ["logs"],
        "config": {
            "learning_rate": 1.0, "optimizer": "sgd",
            "batch_size": 64, "weight_decay": 0.0,
            "architecture": "VGG16", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1, "train_loss": 2.30, "val_loss": 2.34, "lr": 1.0},
            {"epoch": 2, "train_loss": 15.21, "val_loss": 18.44, "lr": 1.0},
            {"epoch": 3, "train_loss": 0.82, "val_loss": 12.10, "lr": 1.0},
            {"epoch": 4, "train_loss": 9.73, "val_loss": 8.67, "lr": 1.0},
            {"epoch": 5, "train_loss": 3.15, "val_loss": 14.22, "lr": 1.0},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "learning_rate_too_high",
        "correct_fix": "reduce learning_rate to 0.01",
        "requires_fix": False,
    },

    "overfitting": {
        "failure_mode": "overfitting",
        "difficulty": "easy",
        "required_sources": ["logs"],
        "config": {
            "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 32, "weight_decay": 0.0, "dropout": 0.0,
            "architecture": "ResNet50", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.10, "val_loss": 2.15, "train_acc": 0.32, "val_acc": 0.30},
            {"epoch": 5,  "train_loss": 1.20, "val_loss": 1.25, "train_acc": 0.61, "val_acc": 0.58},
            {"epoch": 10, "train_loss": 0.45, "val_loss": 1.41, "train_acc": 0.87, "val_acc": 0.60},
            {"epoch": 15, "train_loss": 0.12, "val_loss": 1.89, "train_acc": 0.97, "val_acc": 0.57},
            {"epoch": 20, "train_loss": 0.03, "val_loss": 2.34, "train_acc": 0.99, "val_acc": 0.54},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "overfitting",
        "correct_fix": "add dropout=0.3 and weight_decay=0.01",
        "requires_fix": False,
    },

    "underfitting": {
        "failure_mode": "underfitting",
        "difficulty": "easy",
        "required_sources": ["logs"],
        "config": {
            "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 32, "weight_decay": 0.0,
            "architecture": "LinearClassifier", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.29, "val_loss": 2.30, "train_acc": 0.11, "val_acc": 0.10},
            {"epoch": 5,  "train_loss": 2.24, "val_loss": 2.25, "train_acc": 0.13, "val_acc": 0.12},
            {"epoch": 10, "train_loss": 2.22, "val_loss": 2.22, "train_acc": 0.14, "val_acc": 0.13},
            {"epoch": 20, "train_loss": 2.21, "val_loss": 2.21, "train_acc": 0.14, "val_acc": 0.14},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "underfitting",
        "correct_fix": "increase model capacity or use a deeper architecture",
        "requires_fix": False,
    },

    # ─── MEDIUM — requires correlating logs + config ─────────────────────────

    "learning_rate_too_low": {
        "failure_mode": "learning_rate_too_low",
        "difficulty": "medium",
        "required_sources": ["logs", "config"],
        "config": {
            "learning_rate": 0.000001, "optimizer": "adam",
            "batch_size": 32, "weight_decay": 0.0,
            "architecture": "ResNet18", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.302, "val_loss": 2.305, "lr": 0.000001},
            {"epoch": 5,  "train_loss": 2.298, "val_loss": 2.300, "lr": 0.000001},
            {"epoch": 10, "train_loss": 2.290, "val_loss": 2.293, "lr": 0.000001},
            {"epoch": 20, "train_loss": 2.275, "val_loss": 2.278, "lr": 0.000001},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "learning_rate_too_low",
        "correct_fix": "increase learning_rate to 0.001",
        "requires_fix": False,
    },

    "missing_regularization": {
        "failure_mode": "missing_regularization",
        "difficulty": "medium",
        "required_sources": ["logs", "config"],
        "config": {
            "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 32, "weight_decay": 0.0, "dropout": 0.0,
            "architecture": "ResNet101", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.10, "val_loss": 2.12, "train_acc": 0.33, "val_acc": 0.31},
            {"epoch": 10, "train_loss": 0.30, "val_loss": 1.20, "train_acc": 0.93, "val_acc": 0.62},
            {"epoch": 20, "train_loss": 0.05, "val_loss": 1.85, "train_acc": 0.99, "val_acc": 0.59},
            {"epoch": 30, "train_loss": 0.01, "val_loss": 2.10, "train_acc": 1.00, "val_acc": 0.57},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "missing_regularization",
        "correct_fix": "add weight_decay=0.01 and dropout=0.3",
        "requires_fix": False,
    },

    "batch_size_too_small": {
        "failure_mode": "batch_size_too_small",
        "difficulty": "medium",
        "required_sources": ["logs", "config"],
        "config": {
            "learning_rate": 0.001, "optimizer": "sgd", "momentum": 0.9,
            "batch_size": 2, "weight_decay": 0.0001,
            "architecture": "ResNet18", "dataset": "ImageNet",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 4.12, "val_loss": 3.98, "lr": 0.001},
            {"epoch": 2,  "train_loss": 3.87, "val_loss": 3.91, "lr": 0.001},
            {"epoch": 3,  "train_loss": 4.45, "val_loss": 4.01, "lr": 0.001},
            {"epoch": 4,  "train_loss": 3.21, "val_loss": 3.87, "lr": 0.001},
            {"epoch": 5,  "train_loss": 4.78, "val_loss": 3.95, "lr": 0.001},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "batch_size_too_small",
        "correct_fix": "increase batch_size to at least 32",
        "requires_fix": False,
    },

    "optimizer_misconfiguration": {
        "failure_mode": "optimizer_misconfiguration",
        "difficulty": "medium",
        "required_sources": ["logs", "config"],
        "config": {
            "learning_rate": 0.01, "optimizer": "sgd", "momentum": 0.0,
            "batch_size": 64, "weight_decay": 0.0,
            "architecture": "ResNet18", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.30, "val_loss": 2.31, "lr": 0.01},
            {"epoch": 5,  "train_loss": 2.27, "val_loss": 2.28, "lr": 0.01},
            {"epoch": 10, "train_loss": 2.21, "val_loss": 2.22, "lr": 0.01},
            {"epoch": 20, "train_loss": 2.10, "val_loss": 2.12, "lr": 0.01},
        ],
        "gradient_norms": None,
        "correct_diagnosis": "optimizer_misconfiguration",
        "correct_fix": "set momentum=0.9 for SGD optimizer",
        "requires_fix": False,
    },

    # ─── HARD — requires logs + config + gradients, fix must be provided ────

    "vanishing_gradients": {
        "failure_mode": "vanishing_gradients",
        "difficulty": "hard",
        "required_sources": ["logs", "config", "gradients"],
        "config": {
            "learning_rate": 0.001, "optimizer": "sgd", "momentum": 0.9,
            "batch_size": 32, "activation": "sigmoid",
            "architecture": "DeepMLP_20layers", "dataset": "MNIST",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.30, "val_loss": 2.30, "train_acc": 0.11},
            {"epoch": 5,  "train_loss": 2.29, "val_loss": 2.29, "train_acc": 0.11},
            {"epoch": 10, "train_loss": 2.28, "val_loss": 2.29, "train_acc": 0.12},
            {"epoch": 20, "train_loss": 2.27, "val_loss": 2.28, "train_acc": 0.12},
        ],
        "gradient_norms": [
            {"layer": "output",  "norm": 0.21},
            {"layer": "layer_15","norm": 0.0031},
            {"layer": "layer_10","norm": 0.000042},
            {"layer": "layer_5", "norm": 0.0000003},
            {"layer": "layer_1", "norm": 0.00000001},
        ],
        "correct_diagnosis": "vanishing_gradients",
        "correct_fix": "switch activation to relu and add batch normalization",
        "requires_fix": True,
    },

    "dying_relu": {
        "failure_mode": "dying_relu",
        "difficulty": "hard",
        "required_sources": ["logs", "config", "gradients"],
        "config": {
            "learning_rate": 0.1, "optimizer": "sgd", "momentum": 0.9,
            "batch_size": 32, "activation": "relu",
            "architecture": "DeepMLP_10layers", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1, "train_loss": 2.31, "val_loss": 2.32, "train_acc": 0.10},
            {"epoch": 2, "train_loss": 1.95, "val_loss": 2.01, "train_acc": 0.28},
            {"epoch": 3, "train_loss": 1.95, "val_loss": 2.01, "train_acc": 0.28},
            {"epoch": 5, "train_loss": 1.95, "val_loss": 2.01, "train_acc": 0.28},
        ],
        "gradient_norms": [
            {"layer": "output",   "norm": 0.15},
            {"layer": "layer_8",  "norm": 0.0},
            {"layer": "layer_6",  "norm": 0.0},
            {"layer": "layer_4",  "norm": 0.0},
            {"layer": "layer_2",  "norm": 0.0},
        ],
        "correct_diagnosis": "dying_relu",
        "correct_fix": "reduce learning_rate to 0.001 or switch to leaky_relu activation",
        "requires_fix": True,
    },

    "bad_weight_initialization": {
        "failure_mode": "bad_weight_initialization",
        "difficulty": "hard",
        "required_sources": ["logs", "config", "gradients"],
        "config": {
            "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 32, "weight_init": "normal_std_100",
            "architecture": "ResNet18", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1, "train_loss": float("nan"), "val_loss": float("nan")},
            {"epoch": 2, "train_loss": float("nan"), "val_loss": float("nan")},
        ],
        "gradient_norms": [
            {"layer": "layer_1", "norm": 98432.1},
            {"layer": "layer_2", "norm": 74219.8},
            {"layer": "layer_3", "norm": 55103.4},
        ],
        "correct_diagnosis": "bad_weight_initialization",
        "correct_fix": "use kaiming or xavier weight initialization",
        "requires_fix": True,
    },

    "lr_scheduler_misconfiguration": {
        "failure_mode": "lr_scheduler_misconfiguration",
        "difficulty": "hard",
        "required_sources": ["logs", "config", "gradients"],
        "config": {
            "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 32, "lr_scheduler": "StepLR",
            "step_size": 5, "gamma": 10.0,
            "architecture": "ResNet18", "dataset": "CIFAR-10",
        },
        "logs": [
            {"epoch": 1,  "train_loss": 2.30, "val_loss": 2.32, "lr": 0.001},
            {"epoch": 3,  "train_loss": 1.80, "val_loss": 1.85, "lr": 0.001},
            {"epoch": 5,  "train_loss": 1.20, "val_loss": 1.30, "lr": 0.001},
            {"epoch": 6,  "train_loss": 9.87, "val_loss": 11.20, "lr": 0.01},
            {"epoch": 10, "train_loss": 0.95, "val_loss": 1.05, "lr": 0.01},
            {"epoch": 11, "train_loss": 87.3, "val_loss": 94.1,  "lr": 0.1},
        ],
        "gradient_norms": [
            {"epoch": 5,  "norm": 0.42},
            {"epoch": 6,  "norm": 18.73},
            {"epoch": 10, "norm": 0.38},
            {"epoch": 11, "norm": 156.2},
        ],
        "correct_diagnosis": "lr_scheduler_misconfiguration",
        "correct_fix": "set gamma to 0.1 so the scheduler decreases lr instead of increasing it",
        "requires_fix": True,
    },
}