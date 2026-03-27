#!/usr/bin/env python
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import threading


@dataclass
class TrainingRecord:
    round_num: int
    accuracy: float
    loss: float
    timestamp: str
    params: Dict
    privacy_budget: float


@dataclass
class TrainingSession:
    session_id: str
    dataset_name: str
    algorithm: str
    params: Dict
    start_time: str
    records: List[TrainingRecord] = field(default_factory=list)
    best_accuracy: float = 0.0
    best_round: int = 0
    is_terminated: bool = False
    termination_reason: str = ""
    total_rounds: int = 0


class TrainingMonitor:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.sessions: Dict[str, TrainingSession] = {}
        self.current_session_id: Optional[str] = None
        self.termination_callback: Optional[Callable] = None
        self._lock = threading.Lock()
        
    def start_session(self, dataset_name: str, algorithm: str, 
                      params: Dict, session_id: Optional[str] = None) -> str:
        with self._lock:
            if session_id is None:
                session_id = f"{dataset_name}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            session = TrainingSession(
                session_id=session_id,
                dataset_name=dataset_name,
                algorithm=algorithm,
                params=params,
                start_time=datetime.now().isoformat(),
            )
            self.sessions[session_id] = session
            self.current_session_id = session_id
            
            print(f"\n{'='*60}")
            print(f"[监控器] 新训练会话开始")
            print(f"  会话ID: {session_id}")
            print(f"  数据集: {dataset_name}")
            print(f"  算法: {algorithm}")
            print(f"  参数: {params}")
            print(f"  耐心值: {self.patience}轮")
            print(f"{'='*60}\n")
            
            return session_id
    
    def record_round(self, round_num: int, accuracy: float, loss: float,
                     privacy_budget: float = 0.0, params: Optional[Dict] = None) -> Dict:
        with self._lock:
            if self.current_session_id is None:
                return {"status": "error", "message": "没有活动的训练会话"}
            
            session = self.sessions[self.current_session_id]
            
            record = TrainingRecord(
                round_num=round_num,
                accuracy=accuracy,
                loss=loss,
                timestamp=datetime.now().isoformat(),
                params=params or session.params,
                privacy_budget=privacy_budget
            )
            session.records.append(record)
            session.total_rounds = round_num
            
            if accuracy > session.best_accuracy:
                session.best_accuracy = accuracy
                session.best_round = round_num
            
            should_terminate, reason = self._check_termination(session)
            
            result = {
                "status": "recorded",
                "round": round_num,
                "accuracy": accuracy,
                "best_accuracy": session.best_accuracy,
                "best_round": session.best_round,
                "should_terminate": should_terminate,
                "termination_reason": reason if should_terminate else None,
                "rounds_since_best": round_num - session.best_round
            }
            
            if should_terminate:
                session.is_terminated = True
                session.termination_reason = reason
                print(f"\n[监控器] 训练终止信号: {reason}")
                if self.termination_callback:
                    self.termination_callback(self.current_session_id, reason)
            
            return result
    
    def _check_termination(self, session: TrainingSession) -> tuple:
        if len(session.records) < self.patience:
            return False, ""
        
        recent_records = session.records[-self.patience:]
        best_in_recent = max(r.accuracy for r in recent_records)
        first_in_recent = recent_records[0].accuracy
        
        if best_in_recent - first_in_recent < self.min_delta:
            return True, f"连续{self.patience}轮准确率无显著提升 (阈值: {self.min_delta})"
        
        if all(r.accuracy < session.best_accuracy - self.min_delta for r in recent_records[-2:]):
            return True, f"准确率持续下降，低于最佳值{session.best_accuracy:.4f}"
        
        return False, ""
    
    def end_session(self, session_id: Optional[str] = None) -> Dict:
        with self._lock:
            sid = session_id or self.current_session_id
            if sid is None or sid not in self.sessions:
                return {"status": "error", "message": "会话不存在"}
            
            session = self.sessions[sid]
            duration = self._calculate_duration(session.start_time)
            
            summary = {
                "session_id": sid,
                "dataset": session.dataset_name,
                "algorithm": session.algorithm,
                "total_rounds": session.total_rounds,
                "best_accuracy": session.best_accuracy,
                "best_round": session.best_round,
                "duration_seconds": duration,
                "is_terminated": session.is_terminated,
                "termination_reason": session.termination_reason,
                "final_params": session.params,
                "accuracy_history": [r.accuracy for r in session.records],
                "loss_history": [r.loss for r in session.records],
            }
            
            print(f"\n{'='*60}")
            print(f"[监控器] 训练会话结束")
            print(f"  总轮数: {session.total_rounds}")
            print(f"  最佳准确率: {session.best_accuracy:.4f} (第{session.best_round}轮)")
            print(f"  训练时长: {duration:.1f}秒")
            print(f"  终止原因: {session.termination_reason or '正常完成'}")
            print(f"{'='*60}\n")
            
            return summary
    
    def _calculate_duration(self, start_time: str) -> float:
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.now()
            return (end - start).total_seconds()
        except:
            return 0.0
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Optional[Dict]:
        sid = session_id or self.current_session_id
        if sid is None or sid not in self.sessions:
            return None
        
        session = self.sessions[sid]
        return {
            "session_id": sid,
            "dataset": session.dataset_name,
            "algorithm": session.algorithm,
            "total_rounds": session.total_rounds,
            "best_accuracy": session.best_accuracy,
            "best_round": session.best_round,
            "records_count": len(session.records),
            "is_terminated": session.is_terminated,
        }
    
    def set_termination_callback(self, callback: Callable) -> None:
        self.termination_callback = callback
    
    def export_session_log(self, filepath: str, session_id: Optional[str] = None) -> None:
        sid = session_id or self.current_session_id
        if sid is None or sid not in self.sessions:
            print("无法导出: 会话不存在")
            return
        
        session = self.sessions[sid]
        log_data = {
            "session_info": {
                "session_id": session.session_id,
                "dataset": session.dataset_name,
                "algorithm": session.algorithm,
                "params": session.params,
                "start_time": session.start_time,
                "best_accuracy": session.best_accuracy,
                "best_round": session.best_round,
                "is_terminated": session.is_terminated,
                "termination_reason": session.termination_reason,
            },
            "records": [
                {
                    "round": r.round_num,
                    "accuracy": r.accuracy,
                    "loss": r.loss,
                    "timestamp": r.timestamp,
                    "privacy_budget": r.privacy_budget,
                }
                for r in session.records
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"训练日志已导出: {filepath}")
    
    def get_accuracy_trend(self, session_id: Optional[str] = None, 
                           window_size: int = 5) -> List[float]:
        sid = session_id or self.current_session_id
        if sid is None or sid not in self.sessions:
            return []
        
        records = self.sessions[sid].records
        if len(records) < window_size:
            return [r.accuracy for r in records]
        
        trend = []
        for i in range(len(records)):
            start_idx = max(0, i - window_size + 1)
            window = records[start_idx:i+1]
            trend.append(sum(r.accuracy for r in window) / len(window))
        
        return trend


class AdaptiveMonitor(TrainingMonitor):
    def __init__(self, patience: int = 3, min_delta: float = 0.001,
                 max_rounds: int = 100):
        super().__init__(patience, min_delta)
        self.max_rounds = max_rounds
        self.param_history: List[Dict] = []
        self.best_overall_accuracy: float = 0.0
        self.best_overall_params: Optional[Dict] = None
    
    def record_param_switch(self, old_params: Dict, new_params: Dict, 
                            reason: str) -> None:
        switch_record = {
            "timestamp": datetime.now().isoformat(),
            "old_params": old_params,
            "new_params": new_params,
            "reason": reason,
        }
        self.param_history.append(switch_record)
        
        print(f"\n[自适应监控器] 参数切换")
        print(f"  原因: {reason}")
        print(f"  旧参数: {old_params}")
        print(f"  新参数: {new_params}")
    
    def update_best_overall(self, accuracy: float, params: Dict) -> bool:
        if accuracy > self.best_overall_accuracy:
            self.best_overall_accuracy = accuracy
            self.best_overall_params = params.copy()
            return True
        return False
    
    def get_best_overall(self) -> Dict:
        return {
            "best_accuracy": self.best_overall_accuracy,
            "best_params": self.best_overall_params,
            "total_sessions": len(self.sessions),
            "total_param_switches": len(self.param_history),
        }
    
    def _check_termination(self, session: TrainingSession) -> tuple:
        should_terminate, reason = super()._check_termination(session)
        
        if session.total_rounds >= self.max_rounds:
            return True, f"达到最大轮数限制 ({self.max_rounds})"
        
        return should_terminate, reason


if __name__ == "__main__":
    monitor = TrainingMonitor(patience=3, min_delta=0.001)
    
    session_id = monitor.start_session(
        dataset_name="MNIST",
        algorithm="DPProx",
        params={"epsilon": 5.0, "mu": 0.1}
    )
    
    test_accuracies = [0.5, 0.6, 0.7, 0.75, 0.78, 0.79, 0.791, 0.792, 0.791]
    for i, acc in enumerate(test_accuracies, 1):
        result = monitor.record_round(i, acc, 1.0 - acc)
        print(f"轮次{i}: 准确率={acc:.3f}, 最佳={result['best_accuracy']:.3f}, "
              f"应终止={result['should_terminate']}")
        if result['should_terminate']:
            break
    
    summary = monitor.end_session()
    print(f"\n最终结果: {summary}")
