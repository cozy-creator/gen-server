-- name: GetAPIKey :one
SELECT * FROM api_keys WHERE key_hash = ?;

-- name: CreateAPIKey :one
INSERT INTO api_keys (key_hash) VALUES (?) RETURNING *;

-- name: RevokeAPIKey :exec
UPDATE api_keys SET is_revoked = ? WHERE key_hash = ? RETURNING *;