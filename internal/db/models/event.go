package models

import (
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
	"github.com/vmihailenco/msgpack"
)

type Event struct {
	bun.BaseModel `bun:"table:events,alias:e"`

	ID        uuid.UUID    `bun:",type:uuid,pk"`
	Type      types.EventType    `bun:",notnull"`
	Data      []byte       `bun:",notnull"`
	JobID     uuid.UUID    `bun:",type:uuid,notnull"`
	Job       *Job         `bun:"rel:belongs-to,join:job_id=id"`
	CreatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

func NewEvent(jobID uuid.UUID, eventType types.EventType, data interface{}) *Event {
	encodedData, err := msgpack.Marshal(data)
	if err != nil {
		panic(err)
	}

	return &Event{
		Data:  encodedData,
		JobID: jobID,
		Type:  eventType,
		ID:    uuid.Must(uuid.NewRandom()),
	}
}
