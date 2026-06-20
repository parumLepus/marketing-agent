from typing import Optional, Dict, Any
from langchain.tools import tool


def make_company_context_tools():
    """
    Factory for the three company-context tools, each call getting its own
    fresh store instead of every user sharing one module-level dict. The
    old module-level company_info_store meant one user's stored industry/
    audience/goal could be read back by a completely different user in an
    unrelated conversation - same class of bug as the image generation
    tool's module-level _last_generated_image.
    """
    company_info_store: Dict[str, Any] = {}

    @tool
    def store_company_context(
            industry: Optional[str] = None,
            audience: Optional[str] = None,
            goal: Optional[str] = None,
            stage: Optional[str] = None,
            challenge: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store company context as you learn it naturally from conversation.
        Don't force collection - only store what the user actually mentions.

        Call this sparingly, only when they explicitly describe their situation.
        """

        # Only update fields that are actually provided
        if 'current' not in company_info_store:
            company_info_store['current'] = {}

        if industry:
            company_info_store['current']['industry'] = industry
        if audience:
            company_info_store['current']['audience'] = audience
        if goal:
            company_info_store['current']['goal'] = goal
        if stage:
            company_info_store['current']['stage'] = stage
        if challenge:
            company_info_store['current']['challenge'] = challenge

        return {
            'status': 'Context stored',
            'context': company_info_store['current']
        }

    @tool
    def get_company_context() -> Optional[Dict[str, Any]]:
        """
        Retrieve any company context learned so far from conversation.
        Returns None if nothing has been stored yet.
        """
        return company_info_store.get('current')

    @tool
    def clear_context() -> str:
        """
        Clear stored context (useful for starting a new project discussion).
        """
        company_info_store['current'] = {}
        return "Context cleared."

    return store_company_context, get_company_context, clear_context
