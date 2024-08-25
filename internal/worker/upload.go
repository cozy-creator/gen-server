package worker

import (
	"cozy-creator/gen-server/internal/services"
	"fmt"

	"github.com/gammazero/workerpool"
)

type UploadWorker struct {
	wp       *workerpool.WorkerPool
	uploader services.Uploader
}

func NewUploadWorker(uploader services.Uploader, maxWorkers int) *UploadWorker {
	wp := workerpool.New(maxWorkers)

	return &UploadWorker{
		wp:       wp,
		uploader: uploader,
	}
}

func InitializeUploadWorker(uploader services.Uploader, maxWorkers int) {
	cacheUploadWorker(NewUploadWorker(uploader, maxWorkers))
}

func (w *UploadWorker) Stop() {
	w.wp.Stop()
}

func (w *UploadWorker) Upload(file services.FileMeta, response chan string) {
	upload := func() {
		w.upload(file, response)
	}

	w.wp.Submit(upload)
}

func (w UploadWorker) UploadAndWait(file services.FileMeta, response chan string) {
	upload := func() {
		w.upload(file, response)
	}
	w.wp.SubmitWait(upload)
}

func (w *UploadWorker) upload(file services.FileMeta, response chan string) {
	if w.uploader == nil {
		return
	}

	url, err := w.uploader.Upload(file)
	if err != nil {
		fmt.Println(err)
		return
	}

	response <- url
}
