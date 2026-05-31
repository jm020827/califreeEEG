class MissingAssetError(RuntimeError):
    """Raised when an external data/model asset is not available locally."""


class AssetVerificationError(RuntimeError):
    """Raised when an asset exists but does not satisfy the expected schema."""

