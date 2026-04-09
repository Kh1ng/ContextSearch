# API Reference

## Base URL

All endpoints are relative to `/api/v1`. The full base URL in production is `https://api.projectx.io/api/v1`.

## Authentication

Include the JWT access token in every request:

```
Authorization: Bearer <access_token>
```

Unauthenticated requests return `401 Unauthorized`. Requests with insufficient role return `403 Forbidden`.

## Pagination

List endpoints accept `page` (1-indexed) and `page_size` (default 20, max 100) query parameters. Response envelope:

```json
{
  "items": [...],
  "total": 142,
  "page": 1,
  "page_size": 20,
  "pages": 8
}
```

## Error Format

All errors return a JSON body:

```json
{
  "detail": "Human-readable message",
  "code": "MACHINE_READABLE_CODE"
}
```

Common codes: `NOT_FOUND`, `FORBIDDEN`, `VALIDATION_ERROR`, `RATE_LIMITED`, `INTERNAL_ERROR`.

## Endpoints

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Exchange credentials for access + refresh tokens |
| POST | `/auth/refresh` | Exchange refresh token for new access token |
| POST | `/auth/logout` | Revoke refresh token |

### Users

| Method | Path | Description |
|--------|------|-------------|
| GET | `/users/me` | Get current user profile |
| PATCH | `/users/me` | Update current user profile |
| GET | `/users` | List users in tenant (admin only) |
| POST | `/users/invite` | Invite a new user by email (admin only) |
| DELETE | `/users/{user_id}` | Remove user from tenant (admin only) |

### Items

| Method | Path | Description |
|--------|------|-------------|
| GET | `/items` | List items (paginated, scoped to tenant) |
| POST | `/items` | Create item |
| GET | `/items/{item_id}` | Get item by ID |
| PATCH | `/items/{item_id}` | Update item |
| DELETE | `/items/{item_id}` | Delete item (admin only) |

### Webhooks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/webhooks` | List registered webhooks |
| POST | `/webhooks` | Register a new webhook endpoint |
| DELETE | `/webhooks/{webhook_id}` | Remove a webhook |

Webhook payloads are signed with HMAC-SHA256 using the webhook secret. Verify the `X-ProjectX-Signature` header before processing.

## Rate Limiting

- **Default**: 1000 requests/minute per tenant
- **Auth endpoints**: 10 requests/minute per IP

Rate limit headers are included in every response:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1712345678
```

When exceeded, the response is `429 Too Many Requests` with a `Retry-After` header.

## Versioning

The current stable version is `v1`. Breaking changes will be released as `v2` with a minimum 6-month deprecation window for `v1`.
