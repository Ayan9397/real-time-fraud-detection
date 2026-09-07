from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


class FraudTransactionRequest(BaseModel):
    """
    Flexible transaction request.

    The fraud model was trained using 432 original features.
    Therefore, the API accepts additional transaction fields
    beyond the commonly used core fields.
    """

    model_config = ConfigDict(
        extra="allow"
    )

    TransactionID: int | None = None
    TransactionDT: float
    TransactionAmt: float

    ProductCD: str | None = None

    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None

    addr1: float | None = None
    addr2: float | None = None

    dist1: float | None = None
    dist2: float | None = None

    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    def to_transaction_dict(self) -> Dict[str, Any]:
        """
        Convert the request into a dictionary containing both
        explicitly defined and additional transaction fields.
        """

        return self.model_dump()