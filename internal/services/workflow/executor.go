package workflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/app"
)

func RunProcessor(app *app.App) error {
	ctx := app.Context()
	queue := app.MQ()
	for {
		message, err := queue.Receive(ctx, "workflows")
		if err != nil {
			return err
		}

		messageData, err := queue.GetMessageData(message)
		if err != nil {
			continue
		}

		var workflow Workflow
		if err := json.Unmarshal(messageData, &workflow); err != nil {
			fmt.Println("failed to unmarshal message:", err)
			continue
		}

		errc := NewWorkflowExecutor(&workflow, app).ExecuteAsync()

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

				queue.Publish(ctx, "workflows", []byte("END"))
				return err
			}
		case <-ctx.Done():
			fmt.Println("Workflow executor stopped")
			return nil
		}
	}
}
