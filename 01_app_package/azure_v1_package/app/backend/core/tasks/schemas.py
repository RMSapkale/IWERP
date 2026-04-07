from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FusionNavigation(BaseModel):
    """Schema for navigation tasks in Oracle Fusion."""
    target_page: str = Field(..., description="The name of the target page or screen.")
    module: str = Field(..., description="The Oracle Fusion module (e.g., SCM, Finance).")
    navigation_steps: List[str] = Field(..., description="Ordered list of steps to reach the page.")
    permissions_required: List[str] = Field(default_factory=list, description="Roles or permissions needed.")

class FusionProcedure(BaseModel):
    """Schema for standard operating procedures."""
    title: str = Field(..., description="The title of the procedure.")
    prerequisites: List[str] = Field(default_factory=list)
    steps: List[str] = Field(..., description="Step-by-step instructions.")
    expected_result: str = Field(..., description="What happens if done correctly.")

class TroubleshootingStep(BaseModel):
    symptom: str
    root_cause: Optional[str] = None
    fix: str

class FusionTroubleshoot(BaseModel):
    """Schema for troubleshooting Oracle Fusion issues."""
    issue_id: str
    severity: str = "Medium"
    diagnostic_steps: List[str]
    resolutions: List[TroubleshootingStep]

class FusionIntegration(BaseModel):
    """Schema for technical integration details."""
    integration_type: str = Field(..., description="e.g., REST API, SOAP, FBDI, BIP")
    endpoint_url: Optional[str] = None
    payload_example: Optional[str] = None
    mapping_rules: Optional[Dict[str, str]] = None
    auth_method: str = "OAuth2"
