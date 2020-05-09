# Raised when coarse_net_path is missing and we are trying to use it
class CoarseNetPathMissingException(Exception):
    pass

# Raised when fine_net_path is missing and we are trying to use it
class FineNetPathMissingException(Exception):
    pass

# Raised when classify_net_path is missing and we are trying to use it
class ClassifyNetPathMissingException(Exception):
    pass

# Raised when ClassifyNet is not loaded and we are trying to use it
class ClassifyNetNotLoadedException(Exception):
    pass

# Raised when MinutiaeNet is not loaded and we are trying to use it
class MinutiaeNetNotLoadedException(Exception):
    pass