from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, ForeignKey, Boolean, ForeignKeyConstraint, PrimaryKeyConstraint, Text, Date, Numeric
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    email = Column(String(255), unique=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    name = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    unban_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    reactions = relationship("ContentReaction", back_populates="user")
    user_personas = relationship(
        "UserPersona",
        back_populates="user",
        primaryjoin="User.user_id == UserPersona.user_id" 
    )
    reviews = relationship("Review", back_populates="user")


class ReactionType(str, Enum):
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"

class ReviewStatus(str, Enum):
    COMMON = "COMMON"
    BLIND = "BLIND"
    SPOILER = "SPOILER"

class ContentReaction(Base):
    __tablename__ = "content_reactions"

    content_reaction_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.user_id'), index=True)
    content_id = Column(BigInteger, index=True)
    reaction = Column(SQLEnum(ReactionType, name='reaction_type_enum', create_type=True), nullable=False)
    type = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    user = relationship("User", back_populates="reactions") 

class Persona(Base):
    __tablename__ = "personas"

    persona_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(String(255))
    name = Column(String(50))
    tag = Column(String(255))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    user_personas = relationship("UserPersona", back_populates="persona")

class UserPersona(Base):
    __tablename__ = "user_personas"

    user_id = Column(BigInteger, ForeignKey('users.user_id'), primary_key=True, index=True)
    persona_id = Column(BigInteger, ForeignKey('personas.persona_id'), primary_key=True, index=True)
    score = Column(Float)

    user = relationship("User", back_populates="user_personas")
    persona = relationship("Persona", back_populates="user_personas")

class Review(Base):
    __tablename__ = "reviews"

    review_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.user_id'), index=True)
    content_id = Column(Integer, index=True)
    type = Column(String(50))
    score = Column(Float, nullable=False)
    review_content = Column(String(255), nullable=True)
    like_count = Column(Integer, default=0, nullable=False)
    report_count = Column(Integer, default=0, nullable=False)
    review_status = Column(SQLEnum(ReviewStatus, name='review_status_enum', create_type=True), default=ReviewStatus.COMMON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    user = relationship("User", back_populates="reviews")

class Content(Base):
    __tablename__ = "contents"

    id = Column(BigInteger, primary_key=True)
    type = Column(String(255), primary_key=True)
    title = Column(String(255), nullable=False)
    overview = Column(Text, nullable=True)
    runtime = Column(Integer, nullable=True)
    rating_average = Column(Numeric(38, 2), nullable=True)
    rating_count = Column(BigInteger, nullable=True)
    release_date = Column(Date, nullable=True)
    backdrop_path = Column(String(255), nullable=True)
    poster_path = Column(String(255), nullable=True)

    recommended_by = relationship(
        "ContentRecommendation",
        foreign_keys="[ContentRecommendation.source_content_id, ContentRecommendation.source_content_type]",
        back_populates="source_content"
    )
    recommended_to = relationship(
        "ContentRecommendation",
        foreign_keys="[ContentRecommendation.recommended_content_id, ContentRecommendation.recommended_content_type]",
        back_populates="recommended_content"
    )


class ContentRecommendation(Base):
    __tablename__ = "content_recommendations"

    source_content_id = Column(BigInteger, nullable=False)
    source_content_type = Column(String(255), nullable=False)
    recommended_content_id = Column(BigInteger, nullable=False)
    recommended_content_type = Column(String(255), nullable=False)

    ranking = Column(Integer, nullable=False)
    score = Column(Float(precision=4), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            'source_content_id',
            'source_content_type',
            'recommended_content_id',
            'recommended_content_type'
        ),
        ForeignKeyConstraint(
            ['source_content_id', 'source_content_type'],
            ['contents.id', 'contents.type'],
            ondelete="CASCADE"
        ),
        ForeignKeyConstraint(
            ['recommended_content_id', 'recommended_content_type'],
            ['contents.id', 'contents.type'],
            ondelete="CASCADE"
        ),
    )

    source_content = relationship(
        "Content",
        foreign_keys=[source_content_id, source_content_type],
        back_populates="recommended_by"
    )
    recommended_content = relationship(
        "Content",
        foreign_keys=[recommended_content_id, recommended_content_type],
        back_populates="recommended_to"
    )

class PopularContent(Base):
    __tablename__ = "popular_contents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    content_id = Column(BigInteger, nullable=False)
    content_type = Column(String(255), nullable=False)

    like_count = Column(BigInteger, nullable=True)
    ranked_date = Column(Date, nullable=False)
    ranking = Column(Integer, nullable=False)
    batch_type = Column(String(255), nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ['content_id', 'content_type'],
            ['contents.id', 'contents.type'],
            ondelete="CASCADE"
        ),
    )

    content = relationship(
        "Content",
        primaryjoin="and_(PopularContent.content_id == Content.id, PopularContent.content_type == Content.type)",
        backref="popular_entries"
    )