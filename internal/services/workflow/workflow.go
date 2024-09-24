package workflow

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

type Workflow struct {
	Nodes []Node `json:"nodes"`
}

type Node struct {
	Id      string               `json:"id"`
	Type    string               `json:"type"`
	Inputs  map[string]PortValue `json:"inputs"`
	Outputs map[string]PortValue `json:"outputs"`
}

type PortValue struct {
	Value interface{} `json:"value,omitempty"`
	Ref   *PortRef    `json:"ref,omitempty"`
}

type PortRef struct {
	NodeId     string `json:"node_id"`
	HandleName string `json:"handle_name"`
}

type NodeOutput map[string]interface{}

type WorkflowExecutor struct {
	Workflow *Workflow
	Outputs  sync.Map

	nodesChan  chan *Node
	errorsChan chan error
}

func NewWorkflowExecutor(workflow *Workflow) *WorkflowExecutor {
	totalNodes := len(workflow.Nodes)

	flow := &WorkflowExecutor{
		Workflow: workflow,
		Outputs:  sync.Map{},

		nodesChan:  make(chan *Node, totalNodes),
		errorsChan: make(chan error, totalNodes),
	}

	flow.initializeOutputs()

	return flow
}

func (e *WorkflowExecutor) initializeOutputs() {
	for _, node := range e.Workflow.Nodes {
		if len(node.Outputs) == 0 {
			continue
		}

		value, _ := e.Outputs.LoadOrStore(node.Id, make(NodeOutput))
		output := value.(NodeOutput)

		for name, v := range node.Outputs {
			if _, exists := output[name]; !exists {
				output[name] = v
			}
		}
	}
}

func (e *WorkflowExecutor) ExecuteAsync(ctx context.Context) <-chan error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- e.Execute(ctx)
	}()

	
	return errc
}

func (e *WorkflowExecutor) Execute(ctx context.Context) error {
	var wg sync.WaitGroup
	nodes, err := e.orderNodes()
	if err != nil {
		return err
	}

	defer close(e.nodesChan)
	e.startExecutorWorkers(ctx, &wg, runtime.NumCPU())

	for len(nodes) > 0 {
		for i := 0; i < len(nodes); i++ {
			node := nodes[i]

			if e.isNodeReadyForExecution(&node) {
				wg.Add(1)

				e.nodesChan <- &node
				nodes = append(nodes[:i], nodes[i+1:]...)
				i--
			}
		}

		select {
		case err := <-e.errorsChan:
			return err
		case <-ctx.Done():
			return ctx.Err()
		default:
			break
		}
	}

	wg.Wait()

	for {
		select {
		case err := <-e.errorsChan:
			return err
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
}

func (e *WorkflowExecutor) startExecutorWorkers(ctx context.Context, wg *sync.WaitGroup, numWorkers int) {
	if numWorkers == 0 {
		numWorkers = runtime.NumCPU()
	}

	for i := 0; i < numWorkers; i++ {
		go e.worker(ctx, wg)
	}
}

func (e *WorkflowExecutor) worker(ctx context.Context, wg *sync.WaitGroup) {
	for node := range e.nodesChan {
		inputs, err := e.resolveInputs(node)
		if err != nil {
			e.errorsChan <- err
			wg.Done()
			return
		}

		output, err := executeNode(ctx, node, inputs)
		if err != nil {
			e.errorsChan <- err
			wg.Done()
			return
		}

		e.storeOutput(node.Id, output)
		wg.Done()
	}

	fmt.Println("worker done")
}

func (e *WorkflowExecutor) isNodeReadyForExecution(node *Node) bool {
	for _, input := range node.Inputs {
		if input.Ref != nil {
			dep := e.getNode(input.Ref.NodeId)
			if dep == nil {
				return false
			}

			if !e.hasOutput(dep.Id) {
				return false
			}
		}
	}

	return true
}

func (e *WorkflowExecutor) resolveInputs(node *Node) (map[string]interface{}, error) {
	inputs := make(map[string]interface{})

	for name, input := range node.Inputs {
		if input.Ref != nil {
			dep := e.getNode(input.Ref.NodeId)
			if dep == nil {
				return nil, fmt.Errorf("node not found: %s", input.Ref.NodeId)
			}

			output := e.getOutput(dep.Id)
			outputValue, ok := output[input.Ref.HandleName].(PortValue)
			if !ok {
				return nil, fmt.Errorf("failed to parse output value: %s", input.Ref.HandleName)
			}

			inputs[name] = outputValue.Value
		} else if input.Value != nil {
			inputs[name] = input.Value
		} else {
			return nil, fmt.Errorf("input value not found: %s", name)
		}
	}

	return inputs, nil
}

func (e *WorkflowExecutor) getNode(id string) *Node {
	for _, node := range e.Workflow.Nodes {
		if node.Id == id {
			return &node
		}
	}

	return nil
}

func (e *WorkflowExecutor) orderNodes() ([]Node, error) {
	ordered := make([]Node, 0, len(e.Workflow.Nodes))
	visiting := make(map[string]bool)
	visited := make(map[string]bool)

	var visit func(node *Node) error
	visit = func(node *Node) error {
		if visiting[node.Id] {
			return fmt.Errorf("cycle detected at node: %s", node.Id)
		}

		if visited[node.Id] {
			return nil
		}

		visiting[node.Id] = true

		for _, input := range node.Inputs {
			if input.Ref != nil {
				dep := e.getNode(input.Ref.NodeId)
				if dep == nil {
					return fmt.Errorf("node not found: %s", input.Ref.NodeId)
				}
				if err := visit(dep); err != nil {
					return err
				}
			}
		}

		visiting[node.Id] = false
		visited[node.Id] = true

		ordered = append(ordered, *node)
		return nil
	}

	for i := range e.Workflow.Nodes {
		node := &e.Workflow.Nodes[i]
		if err := visit(node); err != nil {
			return nil, err
		}
	}

	return ordered, nil
}

func executeNode(ctx context.Context, node *Node, inputs map[string]interface{}) (NodeOutput, error) {
	var (
		output = make(map[string]interface{})
		err    error
	)

	// switch node.Type {
	// case "GenerateImage":
	// 	output, err = generationnode.GenerateImage(ctx, inputs)
	// case "SaveImage":
	// 	output, err = imagenode.SaveImage(ctx, inputs)
	// case "LoadImage":
	// 	output, err = imagenode.LoadImage(ctx, inputs)
	// default:
	// 	return nil, fmt.Errorf("unsupported node type: %s", node.Type)
	// }

	fmt.Println("node output:", output)

	return output, err
}

func (e *WorkflowExecutor) getOutput(nodeId string) NodeOutput {
	if output, ok := e.Outputs.Load(nodeId); ok {
		return output.(NodeOutput)
	}

	return nil
}

func (e *WorkflowExecutor) storeOutput(nodeId string, output NodeOutput) {
	value, _ := e.Outputs.LoadOrStore(nodeId, make(NodeOutput))
	storedOutput := value.(NodeOutput)

	for name, v := range output {
		if _, exists := storedOutput[name]; !exists {
			storedOutput[name] = PortValue{Value: v}
		}
	}
}

func (e *WorkflowExecutor) hasOutput(nodeId string) bool {
	_, loaded := e.Outputs.Load(nodeId)
	return loaded
}

// 813549
