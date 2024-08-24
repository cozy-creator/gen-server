package worker

var uploadWorker *UploadWorker

func cacheUploadWorker(worker *UploadWorker) {
	if worker != nil {
		uploadWorker = worker
	}
}

func GetUploadWorker() *UploadWorker {
	if uploadWorker == nil {
		panic("upload worker not initialized")
	}

	return uploadWorker
}
