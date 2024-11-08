package generation

// import (
// 	"fmt"

// 	"github.com/cozy-creator/gen-server/internal/app"
// 	"github.com/cozy-creator/gen-server/internal/types"
// 	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
// )

// func GenerateImageToImageSync(app *app.App, params *types.GenerateParams, sourceImage interface{}, strength float32) (chan types.GenerationResponse, error) {
//     ctx := app.Context()
//     errc := make(chan error, 1)
//     outputc := make(chan types.GenerationResponse)

//     sendResponse := func(urls []string, index int8, currentModel string, status string) {
//         if len(urls) > 0 {
//             outputc <- types.GenerationResponse{
//                 Output: types.GeneratedOutput{
//                     URLs:  urls,
//                     Model: currentModel,
//                 },
//                 Index:  index,
//                 Status: status,
//             }
//         }
//     }

//     go func() {
//         defer func() {
//             close(outputc)
//             close(errc)
//         }()

//         params.SourceImage = sourceImage
//         params.Strength = strength

//         if err := processImageGen(ctx, params, app.Uploader(), app.MQ(), sendResponse); err != nil {
//             errc <- err
//         }
//     }()

//     select {
//     case err := <-errc:
//         return nil, err
//     case <-ctx.Done():
//         return nil, ctx.Err()
//     default:
//         return outputc, nil
//     }
// }

// func GenerateImageToImageAsync(app *app.App, params *types.GenerateParams, sourceImage interface{}, strength float32) {
//     ctx := app.Context()
//     invoke := func(response types.GenerationResponse) {
//         if err := webhookutil.InvokeWithRetries(ctx, params.WebhookUrl, response, MaxWebhookAttempts); err != nil {
//             fmt.Println("Failed to invoke webhook:", err)
//         }
//     }

//     sendResponse := func(urls []string, index int8, currentModel, status string) {
//         response := types.GenerationResponse{
//             Index:  index,
//             Input:  params,
//             Status: status,
//             ID:     params.ID,
//             Output: types.GeneratedOutput{
//                 URLs:  urls,
//                 Model: currentModel,
//             },
//         }

//         invoke(response)
//     }

//     params.SourceImage = sourceImage
//     params.Strength = strength

//     if err := processImageGen(ctx, params, app.Uploader(), app.MQ(), sendResponse); err != nil {
//         invoke(types.GenerationResponse{Status: StatusFailed})
//     }
// }
