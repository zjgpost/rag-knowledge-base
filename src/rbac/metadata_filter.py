"""RBAC metadata filtering for document-level access control."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MetadataFilter:
    """Filter documents based on RBAC metadata rules.

    Example metadata per chunk:
        {"department": "engineering", "access_level": "internal"}

    Example user role:
        {"department": "engineering", "clearance": "internal"}
    """

    def __init__(self, user_role: Optional[Dict[str, Any]] = None):
        self.user_role = user_role or {}

    def is_allowed(self, metadata: Dict[str, Any]) -> bool:
        """Check whether a document chunk is accessible to the user."""
        if not metadata:
            return True

        department = metadata.get("department")
        access_level = metadata.get("access_level", "public")

        user_department = self.user_role.get("department")
        user_clearance = self.user_role.get("clearance", "public")

        # Public documents are always allowed
        if access_level == "public":
            return True

        # Department-specific documents
        if department and user_department:
            if department != user_department and access_level != "public":
                return False

        # Clearance check
        clearance_order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        if clearance_order.get(access_level, 0) > clearance_order.get(user_clearance, 0):
            return False

        return True

    def filter_doc_ids(
        self,
        doc_metadata: List[Dict[str, Any]],
    ) -> List[int]:
        """Return indices of allowed documents."""
        return [i for i, meta in enumerate(doc_metadata) if self.is_allowed(meta)]
