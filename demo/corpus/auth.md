# Authentication & Authorization

## Overview

ProjectX uses JWT-based authentication. Tokens are issued at login and must be included in the `Authorization: Bearer <token>` header on all protected routes.

## JWT Configuration

Tokens are signed with HS256 using the `SECRET_KEY` environment variable. Settings live in `app/config.py`:

```python
class Settings(BaseSettings):
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
```

## Issuing Tokens

Use the `create_access_token` helper in `app/services/auth_service.py`:

```python
from app.services.auth_service import create_access_token

token = create_access_token(
    data={"sub": str(user.id), "tenant_id": str(user.tenant_id)},
    expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
)
```

## Validating Tokens

The `get_current_user` dependency in `app/middleware/auth.py` decodes and validates the token on every request:

```python
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await user_repo.get_by_id(UUID(user_id), db)
    if user is None:
        raise credentials_exception
    return user
```

## Token Refresh

POST `/auth/refresh` accepts a valid refresh token in the request body and returns a new access token. Refresh tokens are stored hashed in the `refresh_tokens` table. A refresh token may only be used once (rotation enforced).

## Role-Based Access Control

Roles are stored on the `User` model: `admin`, `member`, `viewer`. Enforce with the `require_role` dependency:

```python
@router.delete("/{item_id}")
async def delete_item(
    item_id: UUID,
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    ...
```

## Password Hashing

Use `passlib` with bcrypt. Never store plain-text passwords. The `hash_password` and `verify_password` helpers are in `app/services/auth_service.py`.

## Common Mistakes

- Do not set `ACCESS_TOKEN_EXPIRE_MINUTES` too high in production — 15–30 minutes is standard.
- Never log or return the raw JWT in error responses.
- Refresh tokens must be invalidated on logout — call `auth_service.revoke_refresh_token(token_id)`.
