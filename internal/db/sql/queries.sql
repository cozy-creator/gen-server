
-- name: CreateJob :one
INSERT INTO jobs (id, input, created_at, status)
VALUES ($1, $2, NOW(), $3)
RETURNING id, input, created_at, completed_at, status;

-- name: GetJobByID :one
SELECT id, input, created_at, completed_at, status
FROM jobs
WHERE id = $1;

-- name: UpdateJobStatus :exec
UPDATE jobs
SET status = $2,
    completed_at = CASE WHEN $2 = 'COMPLETED' THEN NOW() ELSE completed_at END
WHERE id = $1;

-- name: ListJobs :many
SELECT id, input, created_at, completed_at, status
FROM jobs
ORDER BY created_at DESC
LIMIT $1 OFFSET $2;

-- Nested Query: Job with Associated Images
-- name: GetJobWithImages :one
SELECT 
    j.id,
    j.input,
    j.created_at,
    j.completed_at,
    j.status,
    COALESCE(json_agg(json_build_object(
        'id', i.id,
        'job_id', i.job_id,
        'url', i.url,
        'created_at', i.created_at
    )) FILTER (WHERE i.id IS NOT NULL), '[]'::json) AS images
FROM jobs j
LEFT JOIN images i ON j.id = i.job_id
WHERE j.id = $1
GROUP BY j.id;

-- name: CreateImage :one
INSERT INTO images (id, job_id, url, created_at)
VALUES ($1, $2, $3, NOW())
RETURNING id, job_id, url, created_at;

-- name: GetImageByID :one
SELECT id, job_id, url, created_at
FROM images
WHERE id = $1;

-- name: ListImagesByJobID :many
SELECT id, job_id, url, created_at
FROM images
WHERE job_id = $1
ORDER BY created_at DESC;

-- name: CreateJobMetric :one
INSERT INTO job_metrics (id, job_id, inference_time, created_at)
VALUES ($1, $2, $3, NOW())
RETURNING id, job_id, inference_time, created_at;

-- name: GetJobMetricsByJobID :many
SELECT id, job_id, inference_time, created_at
FROM job_metrics
WHERE job_id = $1
ORDER BY created_at DESC;

-- Additional Queries for Monitoring and Analytics

-- name: CountJobsByStatus :one
SELECT status, COUNT(*)
FROM jobs
GROUP BY status;

-- name: ListFailedJobs :many
SELECT id, input, created_at, completed_at, status
FROM jobs
WHERE status = 'FAILED'
ORDER BY created_at DESC
LIMIT $1 OFFSET $2;
