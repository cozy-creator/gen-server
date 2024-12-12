package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type PipelineDef struct {
    bun.BaseModel `bun:"table:pipeline_defs"`
    ID            uuid.UUID             `bun:",type:uuid,pk"`
    Name          string                `bun:",unique,notnull"`
    Source        string                `bun:",notnull"`
    ClassName     string                `bun:",nullzero"`
    CustomPipeline string               `bun:",nullzero"`
    Metadata      map[string]any        `bun:",type:jsonb"`
    DefaultArgs   map[string]any        `bun:",type:jsonb,nullzero"`
    Components    map[string]any        `bun:",type:jsonb,nullzero"`
    CreatedAt     bun.NullTime          `bun:",nullzero,default:current_timestamp"`
    UpdatedAt     bun.NullTime          `bun:",nullzero,default:current_timestamp"`
}
