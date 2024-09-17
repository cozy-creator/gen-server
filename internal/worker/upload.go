package worker

import (
	"cozy-creator/gen-server/internal/services/filehandler"

	"github.com/gammazero/workerpool"
)

type UploadWorker struct {
	wp       *workerpool.WorkerPool
	uploader filehandler.FileHandler
}

func NewUploadWorker(uploader filehandler.FileHandler, maxWorkers int) *UploadWorker {
	wp := workerpool.New(maxWorkers)

	return &UploadWorker{
		wp:       wp,
		uploader: uploader,
	}
}

func InitializeUploadWorker(uploader filehandler.FileHandler, maxWorkers int) {
	cacheUploadWorker(NewUploadWorker(uploader, maxWorkers))
}

func (w *UploadWorker) Stop() {
	w.wp.Stop()
}

func (w *UploadWorker) Upload(file filehandler.FileInfo, response chan string) {
	upload := func() {
		w.upload(file, response)
	}

	w.wp.Submit(upload)
}

func (w UploadWorker) UploadAndWait(file filehandler.FileInfo, response chan string) {
	upload := func() {
		w.upload(file, response)
	}
	w.wp.SubmitWait(upload)
}

func (w *UploadWorker) upload(file filehandler.FileInfo, response chan string) {
	if w.uploader == nil {
		return
	}

	url, err := w.uploader.Upload(file)
	if err != nil {
		return
	}

	response <- url
}
