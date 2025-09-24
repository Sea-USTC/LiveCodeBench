import json
import re
import subprocess
import time
import requests
import os
import tempfile
import sys
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass
from enum import Enum

from lcb_runner.benchmarks import CodeGenerationProblem
from lcb_runner.lm_styles import LMStyle


def load_code_generation_dataset(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", revision='refs/pr/7', name=release_version)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    dataset = [problem for problem in dataset if problem.question_title == "maximum-amount-of-money-robot-can-earn"]
    # dataset[0].experience=("<experience>\nDynamic Programming Code Generation Reminder:"
    #                         "When generating DP code, pay special attention to these common oversights:"
    #                         "State transition completeness: Ensure you consider ALL possible previous states that can contribute to the current state. If you update k+1 at k, please avoid overwrite k+1 at k+1."
    #                         "Conditional logic accuracy: When translating the thought process into code, be careful with if-else statement nesting and branching logic. Ensure each subcase follows the correct conditional path."
    #                         "Initialization: Every state that cannot be arrived from other states should be properly assigned an initial value."
    #                         "When you think step by step, regularly check the newly modified code to obey the rules above.\n</experience>\n<think>\n")
    # dataset[0].experience=("Follow the Dynamic Programming Code Regulations.")
    dataset[0].experience=("<think>\n")
    print(f"Loaded {len(dataset)} problems")
    return dataset

def get_codeqwen_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    
    if question.starter_code:
        prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    
    prompt += f"<|im_start|>assistant\n"
    prompt += question.experience
    return prompt

def format_prompt_generation(question: CodeGenerationProblem, LanguageModelStyle: LMStyle) -> str:
    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question)}"
        return prompt
    
    raise NotImplementedError(f"LanguageModelStyle {LanguageModelStyle} not implemented")

def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

def get_sample_test_cases(problem: CodeGenerationProblem):
    return {
        "input_output": json.dumps(
            {
                "inputs": [
                    t.input
                    for t in problem.public_test_cases 
                ],
                "outputs": [
                    t.output
                    for t in problem.public_test_cases
                ],
                "fn_name": problem.metadata.get("func_name", None),
            }
        ),
    }





class VLLMServer:
    def __init__(self, model_path: str, port: int = 8000, max_model_len: int = 32768):
        self.model_path = model_path
        self.port = port
        self.max_model_len = max_model_len
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self):
        """启动vllm服务器"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", "0.9"
        ]
        
        print(f"启动VLLM服务器: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
        
        # 等待服务器启动
        print("等待服务器启动...")
        for i in range(60):
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("VLLM服务器已启动")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("服务器启动超时")
        return False
    
    def stop(self):
        """停止vllm服务器"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("VLLM服务器已停止")

class CodeTester:
    """代码测试器，使用LiveCodeBench的评测逻辑"""
    
    @staticmethod
    def evaluate_generations_by_problem(args) -> tuple:
        """评估单个问题的代码生成结果（benchmark测试）"""
        generations, sample, debug, timeout = args[0]
        
        # 解析evaluation sample
        input_output = json.loads(sample['input_output'])
        inputs = input_output['inputs']
        outputs = input_output['outputs']
        fn_name = input_output.get('fn_name')
        
        results = []
        metadata = []
        
        for generation in generations:
            try:
                result, meta = CodeTester._run_single_test(
                    generation, inputs, outputs, timeout, fn_name
                )
                results.append(result)
                metadata.append(meta)
                
            except Exception as e:
                if debug:
                    print(f"评估错误: {e}")
                results.append({"passed": False, "error": str(e)})
                metadata.append({"compilation_error": str(e)})
        
        return results, metadata
    
    @staticmethod
    def _run_single_test(code: str, inputs: List[str], outputs: List[str], 
                        timeout: int, fn_name: str = None) -> tuple:
        """运行单个代码的所有测试用例"""
        
        all_passed = True
        test_results = []
        metadata = {
            "compilation_error": None,
            "runtime_error": None,
            "timeout": False
        }
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # 如果指定了函数名，添加调用逻辑
                if fn_name:
                    test_code = f"{code}\n\n"
                    test_code += "import sys\n"
                    test_code += "input_line = sys.stdin.readline().strip()\n"
                    test_code += f"result = {fn_name}(*eval(input_line))\n"
                    test_code += "print(result)\n"
                else:
                    test_code = code
                
                f.write(test_code)
                temp_file = f.name
            
            # 运行所有测试用例
            for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
                try:
                    process = subprocess.Popen(
                        [sys.executable, temp_file],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    stdout, stderr = process.communicate(input=test_input, timeout=timeout)
                    
                    if process.returncode != 0:
                        test_results.append({
                            "test_case": i,
                            "passed": False,
                            "expected": expected_output,
                            "actual": stderr.strip(),
                            "error": stderr.strip()
                        })
                        all_passed = False
                        metadata["runtime_error"] = stderr.strip()
                    else:
                        actual_output = stdout.strip()
                        passed = actual_output == expected_output.strip()
                        
                        test_results.append({
                            "test_case": i,
                            "passed": passed,
                            "expected": expected_output,
                            "actual": actual_output,
                            "error": "" if passed else f"Output mismatch"
                        })
                        
                        if not passed:
                            all_passed = False
                            
                except subprocess.TimeoutExpired:
                    process.kill()
                    test_results.append({
                        "test_case": i,
                        "passed": False,
                        "expected": expected_output,
                        "actual": "Timeout",
                        "error": "Execution timeout"
                    })
                    all_passed = False
                    metadata["timeout"] = True
                    break
                    
        except Exception as e:
            metadata["compilation_error"] = str(e)
            all_passed = False
            
        finally:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        result = {
            "passed": all_passed,
            "test_results": test_results,
            "passed_count": sum(1 for r in test_results if r["passed"]),
            "total_count": len(test_results)
        }
        
        return result, metadata
    
    @staticmethod
    def evaluate_sample_test_cases(code: str, sample_test_cases: List[Dict], timeout: int = 6) -> Dict[str, Any]:
        """评估样例测试用例（用于多轮修复判断）"""
        if not sample_test_cases:
            return {'test_passed': True, 'test_results': [], 'passed_count': 0, 'total_count': 0}
        
        test_results = []
        all_passed = True
        
        for i, test_case in enumerate(sample_test_cases):
            test_input = test_case['input']
            expected_output = test_case['output']
            
            try:
                # 创建临时文件运行代码
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(input=test_input, timeout=timeout)
                
                if process.returncode != 0:
                    passed = False
                    actual_output = stderr.strip()
                    error = stderr.strip()
                else:
                    actual_output = stdout.strip()
                    passed = actual_output == expected_output.strip()
                    error = "" if passed else "Output mismatch"
                
                test_results.append({
                    'test_case': i,
                    'passed': passed,
                    'expected': expected_output,
                    'actual': actual_output,
                    'error': error,
                    'input': test_input
                })
                
                if not passed:
                    all_passed = False
                    
                os.unlink(temp_file)
                
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_file)
                test_results.append({
                    'test_case': i,
                    'passed': False,
                    'expected': expected_output,
                    'actual': "Timeout",
                    'error': "Execution timeout",
                    'input': test_input
                })
                all_passed = False
                
            except Exception as e:
                if 'temp_file' in locals():
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                test_results.append({
                    'test_case': i,
                    'passed': False,
                    'expected': expected_output,
                    'actual': str(e),
                    'error': str(e),
                    'input': test_input
                })
                all_passed = False
        
        return {
            'test_passed': all_passed,
            'test_results': test_results,
            'passed_count': sum(1 for r in test_results if r['passed']),
            'total_count': len(test_results)
        }

class MultiRoundCodingExperiment:
    def __init__(self, model_path: str, problem: CodeGenerationProblem, max_rounds: int = 3, 
                 language_model_style: LMStyle = LMStyle.CodeQwenInstruct):
        self.vllm = VLLMServer(model_path)
        self.problem = problem
        self.language_model_style = language_model_style
        self.max_rounds = max_rounds
        self.results = []
        
        # 获取benchmark评测用的evaluation sample（包含所有测试用例）
        self.evaluation_sample = problem.get_evaluation_sample()
        
        # 获取样例测试用例（仅公开测试用例，用于多轮修复）
        self.sample_test_cases = get_sample_test_cases(problem)
        
        # 构造初始prompt
        self.initial_prompt = format_prompt_generation(problem, language_model_style)
    
    def generate_response(self, prompt: str, max_tokens: int = 32000) -> str:
        """生成单个响应"""
        response = requests.post(
            f"{self.vllm.base_url}/v1/completions",
            json={
                "model": self.vllm.model_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": None
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            raise Exception(f"API调用失败: {response.status_code}")
    
    def generate_parallel_responses(self, prompt: str, num_responses: int = 10) -> List[str]:
        """并行生成多个响应"""
        responses = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.generate_response, prompt) for _ in range(num_responses)]
            
            for future in as_completed(futures):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    print(f"生成响应失败: {e}")
                    responses.append("")
        
        return responses
    
    def evaluate_code_benchmark(self, code: str) -> Dict[str, Any]:
        """使用benchmark测试用例评估代码（LiveCodeBench完整评测逻辑）"""
        args = ([[code]], self.evaluation_sample, False, 6)
        
        try:
            results, metadata = CodeTester.evaluate_generations_by_problem((args, 0))
            return {
                'test_passed': results[0]['passed'] if results else False,
                'test_results': results[0] if results else {},
                'metadata': metadata[0] if metadata else {}
            }
        except Exception as e:
            return {
                'test_passed': False,
                'test_results': {'error': str(e), 'passed': False},
                'metadata': {'compilation_error': str(e)}
            }
    
    def evaluate_code_samples(self, code: str) -> Dict[str, Any]:
        """使用样例测试用例评估代码（用于多轮修复判断）"""
        return CodeTester.evaluate_sample_test_cases(code, self.sample_test_cases)
    
    def create_repair_prompt(self, original_code: str, sample_test_result: Dict) -> str:
        """基于样例测试结果创建修复prompt"""
        error_info = []
        
        if 'test_results' in sample_test_result and sample_test_result['test_results']:
            for result in sample_test_result['test_results']:
                if not result['passed']:
                    error_info.append(f"样例测试用例 {result['test_case'] + 1}:")
                    error_info.append(f"  输入: {result['input']}")
                    error_info.append(f"  期望输出: {result['expected']}")
                    error_info.append(f"  实际输出: {result['actual']}")
                    if result['error']:
                        error_info.append(f"  错误信息: {result['error']}")
                    error_info.append("")
        
        repair_prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user

原始问题:
{self.problem.question_content}

之前的代码没有通过样例测试：
```python
{original_code}
```

样例测试失败信息:
{chr(10).join(error_info)}

请分析错误原因，并生成修复后的完整代码。请确保代码能够正确处理所有样例测试用例。

```python
# YOUR FIXED CODE HERE
```

<|im_end|>
<|im_start|>assistant
<think>
"""
        return repair_prompt
    
    def _get_sample_error_summary(self, sample_eval_result: Dict) -> str:
        """获取样例测试错误摘要"""
        if not sample_eval_result.get('test_results'):
            return "未知错误"
        
        test_results = sample_eval_result['test_results']
        if isinstance(test_results, list):
            failed_tests = [r for r in test_results if not r['passed']]
            if failed_tests:
                error_types = set()
                for failed in failed_tests:
                    if failed['error'] and 'timeout' in failed['error'].lower():
                        error_types.add("超时")
                    elif failed['error'] and failed['error'] != "Output mismatch":
                        error_types.add("运行时错误")
                    else:
                        error_types.add("输出不匹配")
                return f"样例测试失败: {len(failed_tests)}个用例, " + ", ".join(error_types)
        
        return "样例测试失败"
    
    def _get_benchmark_error_summary(self, benchmark_eval_result: Dict) -> str:
        """获取benchmark测试错误摘要"""
        metadata = benchmark_eval_result.get('metadata', {})
        
        if metadata.get('compilation_error'):
            return f"编译错误: {metadata['compilation_error'][:100]}..."
        if metadata.get('runtime_error'):
            return f"运行时错误: {metadata['runtime_error'][:100]}..."
        if metadata.get('timeout'):
            return "执行超时"
        
        test_results = benchmark_eval_result.get('test_results', {})
        if test_results.get('test_results'):
            failed_tests = [r for r in test_results['test_results'] if not r['passed']]
            if failed_tests:
                return f"Benchmark测试失败: {len(failed_tests)}个用例未通过"
        
        return "未知错误"
    
    def run_experiment(self):
        """运行完整实验"""
        if not self.vllm.start():
            print("启动VLLM服务器失败")
            return
        
        try:
            for round_num in range(1, self.max_rounds + 1):
                print(f"\n=== Round {round_num} ===")
                
                if round_num == 1:
                    # 第一轮：初始代码生成
                    prompt = self.initial_prompt
                    responses = self.generate_parallel_responses(prompt, num_responses=10)
                else:
                    # 后续轮：修复代码（基于样例测试结果）
                    responses = []
                    for i, prev_result in enumerate(self.results[-1]):
                        if not prev_result['sample_test_passed']:
                            repair_prompt = self.create_repair_prompt(
                                prev_result['code'], 
                                prev_result['sample_test_results']
                            )
                            response = self.generate_response(repair_prompt)
                            responses.append(response)
                        else:
                            # 样例测试已通过的代码不需要修复
                            responses.append(prev_result['raw_response'])
                
                # 处理响应
                round_results = []
                for i, response in enumerate(responses):
                    print(f"Processing Response {i+1}/{len(responses)}")
                    
                    # 提取代码
                    code = extract_code(response)
                    
                    if not code:
                        print(f"Response {i+1} valid code not found.")
                        round_results.append({
                            'response_id': i,
                            'raw_response': response,
                            'code': '',
                            'sample_test_passed': False,
                            'sample_test_results': {},
                            'benchmark_test_passed': False,
                            'benchmark_test_results': {},
                            'sample_error_reason': 'code not found',
                            'benchmark_error_reason': 'code not found'
                        })
                        continue
                    
                    # 1. 先用样例测试用例评估（用于判断是否需要下一轮修复）
                    sample_eval_result = self.evaluate_code_samples(code)
                    
                    # 2. 再用benchmark测试用例评估（最终评分标准）
                    benchmark_eval_result = self.evaluate_code_benchmark(code)
                    
                    round_results.append({
                        'response_id': i,
                        'raw_response': response,
                        'code': code,
                        'sample_test_passed': sample_eval_result['test_passed'],
                        'sample_test_results': sample_eval_result,
                        'benchmark_test_passed': benchmark_eval_result['test_passed'],
                        'benchmark_test_results': benchmark_eval_result,
                        'sample_error_reason': '' if sample_eval_result['test_passed'] else self._get_sample_error_summary(sample_eval_result),
                        'benchmark_error_reason': '' if benchmark_eval_result['test_passed'] else self._get_benchmark_error_summary(benchmark_eval_result)
                    })
                    
                    # 打印样例测试和benchmark测试结果
                    sample_passed_count = sample_eval_result.get('passed_count', 0)
                    sample_total_count = sample_eval_result.get('total_count', 0)
                    benchmark_passed = benchmark_eval_result['test_passed']
                    benchmark_passed_count = benchmark_eval_result['test_results'].get('passed_count', 0)
                    benchmark_total_count = benchmark_eval_result['test_results'].get('total_count', 0)
                    
                    print(f"响应 {i+1}: 样例 {'通过' if sample_eval_result['test_passed'] else '失败'} "
                          f"({sample_passed_count}/{sample_total_count}), "
                          f"Benchmark {'通过' if benchmark_passed else '失败'} "
                          f"({benchmark_passed_count}/{benchmark_total_count})")
                
                self.results.append(round_results)
                
                # 统计本轮结果
                sample_passed_count = sum(1 for r in round_results if r['sample_test_passed'])
                benchmark_passed_count = sum(1 for r in round_results if r['benchmark_test_passed'])
                
                print(f"第 {round_num} 轮结果:")
                print(f"  样例测试通过率: {sample_passed_count}/{len(round_results)} ({sample_passed_count/len(round_results)*100:.1f}%)")
                print(f"  Benchmark通过率: {benchmark_passed_count}/{len(round_results)} ({benchmark_passed_count/len(round_results)*100:.1f}%)")
                
                # 如果所有响应的样例测试都通过了，提前结束
                if sample_passed_count == len(round_results):
                    print("所有响应的样例测试都已通过，实验结束")
                    break
        
        finally:
            self.vllm.stop()
    
    def save_results(self, filename: str = "experiment_results.json"):
        """保存实验结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {filename}")
    
    def analyze_results(self):
        """分析实验结果"""
        print("\n=== 实验结果分析 ===")
        
        for round_num, round_results in enumerate(self.results, 1):
            sample_passed_count = sum(1 for r in round_results if r['sample_test_passed'])
            benchmark_passed_count = sum(1 for r in round_results if r['benchmark_test_passed'])
            total_count = len(round_results)
            
            print(f"第 {round_num} 轮:")
            print(f"  样例测试通过率: {sample_passed_count}/{total_count} ({sample_passed_count/total_count*100:.1f}%)")
            print(f"  Benchmark通过率: {benchmark_passed_count}/{total_count} ({benchmark_passed_count/total_count*100:.1f}%)")
            
            # 样例测试错误统计
            sample_error_types = {}
            benchmark_error_types = {}
            
            for result in round_results:
                if not result['sample_test_passed'] and result['sample_error_reason']:
                    error_reason = result['sample_error_reason']
                    sample_error_types[error_reason] = sample_error_types.get(error_reason, 0) + 1
                
                if not result['benchmark_test_passed'] and result['benchmark_error_reason']:
                    benchmark_error_reason = result['benchmark_error_reason']
                    benchmark_error_types[benchmark_error_reason] = benchmark_error_types.get(benchmark_error_reason, 0) + 1
            
            if sample_error_types:
                print("  样例测试错误类型:")
                for error, count in sample_error_types.items():
                    print(f"    {error}: {count}")
            
            if benchmark_error_types:
                print("  Benchmark错误类型:")
                for error, count in benchmark_error_types.items():
                    print(f"    {error}: {count}")
            
            print()

if __name__ == "__main__":
    
    dataset = load_code_generation_dataset("v6")
    
    # 选择一个问题进行测试（或者可以遍历所有问题）
    test_problem = dataset[0]  # 使用第一个问题作为示例
    
    print(f"Question Title: {test_problem.question_title}")
    print(f"Question ID: {test_problem.question_id}")
    print(f"Benchmark testcases num: {len(test_problem.public_test_cases + test_problem.private_test_cases)}")
    print(f"  - public: {len(test_problem.public_test_cases)}")
    print(f"  - private: {len(test_problem.private_test_cases)}")
    
    # 显示样例测试用例
    sample_cases = get_sample_test_cases(test_problem)
    if sample_cases:
        print("Public preview(first 2):")
        for i, case in enumerate(sample_cases[:2]):  # 只显示前2个
            print(f"  Case {i+1}:")
            print(f"    Input: {case['input'][:100]}...")  # 截断显示
            print(f"    Output: {case['output'][:100]}...")
        if len(sample_cases) > 2:
            print(f"  ... omit {len(sample_cases) - 2} cases")
        print()
    
    # 启动实验
    model_path = "your-model-path"  # 替换为你的模型路径
    
    experiment = MultiRoundCodingExperiment(
        model_path=model_path,
        problem=test_problem,
        max_rounds=3,
        language_model_style=LMStyle.CodeQwenInstruct
    )
    
    print("开始运行多轮代码生成实验...")
    print("=" * 60)
    
    # 运行实验
    experiment.run_experiment()
    experiment.analyze_results()
    
    # 保存结果
    result_filename = f"experiment_results_{test_problem.question_title.replace('-', '_')}.json"
    experiment.save_results(result_filename)
    
    print(f"实验完成！结果已保存到 {result_filename}")
    
    # 如果要批量测试多个问题，可以使用以下代码:
    """
    # 批量测试示例
    for i, problem in enumerate(dataset[:5]):  # 测试前5个问题
        print(f"\\n测试问题 {i+1}/{len(dataset[:5])}: {problem.question_title}")
        
        experiment = MultiRoundCodingExperiment(
            model_path=model_path,
            problem=problem,
            max_rounds=3,
            language_model_style=LMStyle.CodeQwenInstruct
        )
        
        experiment.run_experiment()
        experiment.analyze_results()
        experiment.save_results(f"batch_results_{i+1}_{problem.question_title.replace('-', '_')}.json")
    """