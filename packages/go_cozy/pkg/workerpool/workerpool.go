package workerpool

import (
	"fmt"
	"sync"
)

type WorkerPool struct {
	maxWorkers int
	stopped    bool
	waiting    []func()
	active     chan func()
	tasks      chan func()
	mu         sync.Mutex
}

func NewWorkerPool(maxWorkers int, autoStart bool) *WorkerPool {
	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	wp := &WorkerPool{
		waiting:    []func(){},
		maxWorkers: maxWorkers,
		tasks:      make(chan func()),
		active:     make(chan func()),
	}

	if autoStart {
		go wp.start()
	}

	return wp
}

func (wp *WorkerPool) start() {
	var workerCount int
	var wg sync.WaitGroup

	for {
		if len(wp.waiting) != 0 {
			if err := wp.processWaiting(); err != nil {
				break
			}
			continue
		}

		select {
		case task, ok := <-wp.tasks:
			if !ok {
				break
			}
			select {
			case wp.active <- task:
			default:
				if workerCount < wp.maxWorkers {
					wg.Add(1)
					go worker(task, wp.active, &wg)
					workerCount++
				} else {
					wp.waiting = append(wp.waiting, task)
				}
			}
			// TODO: handle timeout
		}
	}

	wg.Wait()
}

func (wp *WorkerPool) processWaiting() error {
	select {
	case task, ok := <-wp.tasks:
		if !ok {
			return fmt.Errorf("worker pool closed")
		}
		wp.waiting = append(wp.waiting, task)
	case wp.active <- wp.waiting[0]:
		wp.waiting = wp.waiting[1:]
	}
	return nil
}

func (wp *WorkerPool) Stop() {
	wp.mu.Lock()
	defer wp.mu.Unlock()
	wp.stopped = true
}

func (wp *WorkerPool) Submit(task func()) {
	if task != nil {
		wp.tasks <- task
	}
}

func (wp *WorkerPool) SubmitAndWait(task func()) {
	if task == nil {
		return
	}

	done := make(chan struct{})
	task = func() {
		task()
		close(done)
	}

	wp.Submit(task)
	<-done
}

func (wp *WorkerPool) Start() {
	go wp.start()
}

func worker(task func(), active chan func(), wg *sync.WaitGroup) {
	defer wg.Done()
	for task != nil {
		task()
		task = <-active
	}
}
