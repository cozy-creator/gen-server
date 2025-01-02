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
    PromptDefID   *uuid.UUID            `bun:",nullzero"`
    PromptDef     *PromptDef            `bun:"rel:belongs-to,join:prompt_def_id=id"`
    CreatedAt     bun.NullTime          `bun:",nullzero,default:current_timestamp"`
    UpdatedAt     bun.NullTime          `bun:",nullzero,default:current_timestamp"`
}

type PromptDef struct {
	bun.BaseModel `bun:"table:prompt_defs"`
	ID            uuid.UUID `bun:",type:uuid,pk"`
	PositivePrompt string    `bun:",notnull"`
	NegativePrompt string    `bun:",notnull"`
	CreatedAt      bun.NullTime `bun:",nullzero,default:current_timestamp"`
	UpdatedAt      bun.NullTime `bun:",nullzero,default:current_timestamp"`
}
