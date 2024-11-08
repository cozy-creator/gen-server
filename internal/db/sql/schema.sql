-- Enum for job statuses
CREATE TYPE job_status_enum AS ENUM (
    'IN_QUEUE',
    'IN_PROGRESS',
    'COMPLETED',
    'FAILED'
);

-- Jobs Table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    input JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status job_status_enum NOT NULL
);

-- Images Table with job_id as UUID and foreign key reference to jobs
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    url TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
);

-- Index for faster lookup by image id and job_id
CREATE INDEX IF NOT EXISTS idx_image_job ON images (id, job_id);

-- Job Metrics Table with job_id as UUID and foreign key reference to jobs
CREATE TABLE IF NOT EXISTS job_metrics (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    inference_time FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
);
