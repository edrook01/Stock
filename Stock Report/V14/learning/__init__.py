"""Learning and adaptive feedback module for Stock Analyzer V14"""

# Handle relative imports gracefully - try relative first, fallback to absolute
try:
    from .continuous_service import get_continuous_learning_service, ContinuousLearningService
except ImportError:
    # Fallback for direct execution or test context
    try:
        from learning.continuous_service import get_continuous_learning_service, ContinuousLearningService
    except ImportError:
        # If still fails, set to None to prevent cascading errors
        get_continuous_learning_service = None
        ContinuousLearningService = None

try:
    from .model_updater import get_model_updater, ModelUpdater
except ImportError:
    try:
        from learning.model_updater import get_model_updater, ModelUpdater
    except ImportError:
        get_model_updater = None
        ModelUpdater = None

try:
    from .trade_tracker import get_trade_tracker, TradeTracker, TradeOutcome
except ImportError:
    try:
        from learning.trade_tracker import get_trade_tracker, TradeTracker, TradeOutcome
    except ImportError:
        get_trade_tracker = None
        TradeTracker = None
        TradeOutcome = None

try:
    from .feedback_loop import get_feedback_loop, FeedbackLoop
except ImportError:
    try:
        from learning.feedback_loop import get_feedback_loop, FeedbackLoop
    except ImportError:
        get_feedback_loop = None
        FeedbackLoop = None

try:
    from .prediction_monitor import PredictionMonitor
except ImportError:
    try:
        from learning.prediction_monitor import PredictionMonitor
    except ImportError:
        PredictionMonitor = None

try:
    from .failure_tracker import FailureTracker
except ImportError:
    try:
        from learning.failure_tracker import FailureTracker
    except ImportError:
        FailureTracker = None

try:
    from .diagnostic import LearningDiagnostic
except ImportError:
    try:
        from learning.diagnostic import LearningDiagnostic
    except ImportError:
        LearningDiagnostic = None

__all__ = [
    'get_continuous_learning_service',
    'ContinuousLearningService',
    'get_model_updater',
    'ModelUpdater',
    'get_trade_tracker',
    'TradeTracker',
    'TradeOutcome',
    'get_feedback_loop',
    'FeedbackLoop',
    'PredictionMonitor',
    'FailureTracker',
    'LearningDiagnostic'
]

