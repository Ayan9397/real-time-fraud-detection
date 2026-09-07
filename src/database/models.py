from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


# ---------------------------------------------------------
# Transactions
# ---------------------------------------------------------

class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[int] = mapped_column(
        BigInteger,
        unique=True,
        nullable=False,
        index=False,
    )

    transaction_dt: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    transaction_amt: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
    )

    product_cd: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    card1: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    card2: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    card3: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    card4: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    card5: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    card6: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    addr1: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    addr2: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    dist1: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
    )

    dist2: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
    )

    p_emaildomain: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    r_emaildomain: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    actual_fraud: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )

    predictions: Mapped[list["FraudPrediction"]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------
# Fraud predictions
# ---------------------------------------------------------

class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "transactions.transaction_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    fraud_probability: Mapped[Decimal] = mapped_column(
        Numeric(12, 10),
        nullable=False,
    )

    fraud_prediction: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    decision: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    threshold: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
    )

    prediction_latency_ms: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )

    transaction: Mapped[Transaction] = relationship(
        back_populates="predictions",
    )