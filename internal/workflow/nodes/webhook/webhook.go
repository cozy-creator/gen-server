package webhooknode

import (
	"fmt"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
)

func InvokeWebhook(app *app.App, input map[string]interface{}) (map[string]interface{}, error) {
	targetUrl := input["target_url"].(string)
	retry := (input["retry_attempts"].(float64))
	data := map[string]interface{}{"data": input["data"]}

	headers := make(map[string]string)
	for key, value := range input["headers"].(map[string]interface{}) {
		if str, ok := value.(string); ok {
			headers[key] = str
		} else {
			return nil, fmt.Errorf("invalid header value type: %T", value)
		}
	}

	if _, ok := headers["Content-Type"]; !ok {
		headers["Content-Type"] = "application/json"
	}
	ctx := app.Context()

	// TODO: forward headers to webhookutil.InvokeWithRetries
	if err := webhookutil.InvokeWithRetries(ctx, targetUrl, data, int(retry)); err != nil {
		return nil, err
	}

	return nil, nil
}
