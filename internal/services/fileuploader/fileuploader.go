package fileuploader

import (
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/gammazero/workerpool"
)

type Uploader struct {
	wp          *workerpool.WorkerPool
	filestorage filestorage.FileStorage
}

func NewFileUploader(filestorage filestorage.FileStorage, maxWorkers int) *Uploader {
	wp := workerpool.New(maxWorkers)

	return &Uploader{
		wp:          wp,
		filestorage: filestorage,
	}
}

func (w *Uploader) Stop() {
	w.wp.Stop()
}

func (w *Uploader) Upload(file filestorage.FileInfo, response chan string) {
	upload := func() {
		w.upload(file, response)
	}

	w.wp.Submit(upload)
}

func (w *Uploader) UploadBytes(file []byte, extension string, response chan string) {
	fileHash := hashutil.Blake3Hash(file)
	fileInfo := filestorage.FileInfo{
		Name:      fileHash,
		Extension: extension,
		Content:   file,
		IsTemp:    false,
	}

	w.Upload(fileInfo, response)
}

func (w *Uploader) upload(file filestorage.FileInfo, response chan string) {
	if w.filestorage == nil {
		return
	}

	url, err := w.filestorage.Upload(file)
	if err != nil {
		return
	}

	response <- url
}
