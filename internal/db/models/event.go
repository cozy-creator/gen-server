package models

import (
	"encoding/json"

	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type Event struct {
	bun.BaseModel `bun:"table:events,alias:e"`

	ID        uuid.UUID    		`bun:",type:uuid,pk"`
	Type      types.EventType   `bun:",notnull"`
	Data      []byte       		`bun:"type:jsonb,notnull"`
	JobID     uuid.UUID    		`bun:",type:uuid,notnull"`
	Job       *Job         		`bun:"rel:belongs-to,join:job_id=id"`
	CreatedAt bun.NullTime 		`bun:",nullzero,notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime 		`bun:",nullzero,notnull,default:current_timestamp"`
}

func NewEvent(jobID uuid.UUID, eventType types.EventType, data interface{}) *Event {
	encodedData, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}

	return &Event{
		ID:    uuid.Must(uuid.NewRandom()),
		Type:  eventType,
		Data:  encodedData,
		JobID: jobID,
	}
}
