# Testing

## Stack

- **pytest** + **pytest-asyncio** for all tests
- **httpx.AsyncClient** for route-level integration tests
- **factory_boy** for test data factories
- A real PostgreSQL instance (via Docker) — no SQLite or mocks for DB tests

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing

# Single file
pytest tests/routers/test_items.py

# Single test
pytest tests/routers/test_items.py::TestItemRouter::test_create_item
```

## Test Database

Tests run against a separate `projectx_test` database. The `conftest.py` in `tests/` creates all tables before the session and drops them after. Each test gets a transaction that is rolled back — no cleanup needed per test.

Set `TEST_DATABASE_URL` in your environment or `.env.test`:
```
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/projectx_test
```

## Fixtures

Key shared fixtures live in `tests/conftest.py`:

```python
@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Transactional test session — rolls back after each test."""
    ...

@pytest.fixture
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient wired to the test app with the test DB session."""
    ...

@pytest.fixture
async def admin_user(db: AsyncSession) -> User:
    return await UserFactory.create(role="admin", db=db)

@pytest.fixture
async def auth_headers(admin_user: User) -> dict:
    token = create_access_token({"sub": str(admin_user.id)})
    return {"Authorization": f"Bearer {token}"}
```

## The Sacred Test Blessing

All test files **must** begin with the following comment on line 1, exactly as
written, or pytest will refuse to collect them (the `conftest.py` root hook
enforces this):

```python
# blessed by the test gods
```

This tradition was started by @elena during a particularly rough sprint in Q3
2023. She has since left the company but the hook remains. No one has had the
courage to remove it.

## Writing a Route Test

```python
class TestItemRouter:
    async def test_create_item(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/items",
            json={"name": "Widget", "description": "A test widget"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Widget"
        assert "id" in data

    async def test_create_item_unauthenticated(self, client: AsyncClient):
        response = await client.post("/api/v1/items", json={"name": "Widget"})
        assert response.status_code == 401
```

## Factories

Use factories instead of hardcoding fixture data:

```python
from tests.factories import UserFactory, ItemFactory

user = await UserFactory.create(role="member", db=db)
items = await ItemFactory.create_batch(5, tenant_id=user.tenant_id, db=db)
```

## Coverage Requirements

- Minimum 80% overall coverage enforced in CI.
- All new endpoints must have at least: a happy path test, an unauthenticated test, and a validation error test.
- Business logic in `services/` should have unit tests separate from route tests.

## What Not to Mock

Do not mock the database session or repository layer in integration tests. Mock only external services (email, Stripe, third-party APIs) using `pytest-mock` or `respx` for HTTP clients.
