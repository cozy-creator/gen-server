package workflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/mq"
)

func StartWorkflowExecutor(ctx context.Context, queue mq.MQ) error {
	for {
		message, err := queue.Receive(ctx, "workflows")
		if err != nil {
			return err
		}

		var workflow Workflow
		if err := json.Unmarshal(message, &workflow); err != nil {
			fmt.Println("failed to unmarshal message:", err)
			continue
		}

		errc := NewWorkflowExecutor(&workflow).ExecuteAsync(ctx)

		select {
		case err := <-errc:
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					fmt.Println("Workflow execution timed out", err)
				} else if errors.Is(err, context.Canceled) {
					fmt.Println("Workflow execution was cancelled", err)
				} else {
					fmt.Println("Error executing workflow", err)
				}

				queue.CloseTopic("workflows")
				return err
			}
		case <-ctx.Done():
			fmt.Println("Workflow executor stopped")
			return nil
		}
	}
}
