package worker

import (
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/pkg/workerpool"
)

type UploadWorker struct {
	wp       *workerpool.WorkerPool
	uploader services.Uploader
}

func NewUploadWorker(uploader services.Uploader, maxWorkers int) *UploadWorker {
	wp := workerpool.NewWorkerPool(maxWorkers, false)
	return &UploadWorker{
		wp:       wp,
		uploader: uploader,
	}
}

func (w *UploadWorker) Start() {
	w.wp.Start()
}

func (w *UploadWorker) Upload(file services.FileMeta, response chan string) {
	if w.uploader == nil {
		return
	}

	upload := func() {
		url, err := w.uploader.Upload(file)
		if err != nil {
			return
		}

		response <- url
	}

	w.wp.Submit(upload)
}
