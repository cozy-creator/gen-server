package worker

var uploadWorker *UploadWorker

func SetUploadWorker(worker *UploadWorker) {
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
