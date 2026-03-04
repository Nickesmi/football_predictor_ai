import React from 'react';
import { Loader2, AlertCircle, BarChart3, TrendingUp, Info, CheckCircle2, ShieldAlert, Target, Zap, Trophy, Shield, Crosshair, Flame, Brain, Cpu, Activity, Sigma, CornerUpRight, CreditCard } from 'lucide-react';

/* ── Helpers ──────────────────────────────────────── */

const getConfBadge = (conf) => {
  const map = {
    "Very High": { icon: <CheckCircle2 className="w-3.5 h-3.5" />, cls: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20" },
    "High":      { icon: <TrendingUp className="w-3.5 h-3.5" />,   cls: "text-green-400 bg-green-400/10 border-green-400/20" },
    "Medium":    { icon: <BarChart3 className="w-3.5 h-3.5" />,    cls: "text-yellow-500 bg-yellow-500/10 border-yellow-500/20" },
    "Low":       { icon: <Info className="w-3.5 h-3.5" />,         cls: "text-orange-400 bg-orange-400/10 border-orange-400/20" },
    "Very Low":  { icon: <ShieldAlert className="w-3.5 h-3.5" />,  cls: "text-red-400 bg-red-400/10 border-red-400/20" },
  };
  return map[conf] || map["Medium"];
};

const getBarColor = (pct) => {
  if (pct >= 75) return "from-emerald-500 to-emerald-400";
  if (pct >= 60) return "from-green-500 to-green-400";
  if (pct >= 45) return "from-yellow-500 to-yellow-400";
  if (pct >= 30) return "from-orange-500 to-orange-400";
  return "from-red-500 to-red-400";
};

const getStrengthColor = (v) => {
  if (v >= 1.3) return "text-emerald-400";
  if (v >= 1.0) return "text-yellow-400";
  return "text-red-400";
};

const getVerdictStyle = (verdict) => {
  if (verdict === "Best Choice") return { cls: "text-amber-300 bg-amber-400/15 border-amber-400/30", icon: <Trophy className="w-3.5 h-3.5" /> };
  if (verdict === "Value") return { cls: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20", icon: <Flame className="w-3.5 h-3.5" /> };
  if (verdict === "Fair") return { cls: "text-slate-400 bg-slate-400/10 border-slate-400/20", icon: <BarChart3 className="w-3.5 h-3.5" /> };
  return { cls: "text-red-400 bg-red-400/10 border-red-400/20", icon: <ShieldAlert className="w-3.5 h-3.5" /> };
};

const getPillarIcon = (pillar) => {
  if (pillar === "Corners") return <CornerUpRight className="w-3 h-3" />;
  if (pillar === "Cards") return <CreditCard className="w-3 h-3" />;
  return <Target className="w-3 h-3" />;
};


/* ── Main Component ──────────────────────────────────── */

const MatchDetail = ({ fixture, analysis, loading, error }) => {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <Loader2 className="w-10 h-10 animate-spin text-purple-500 mb-4" />
        <p className="text-purple-400/60 text-xs tracking-[0.2em] uppercase animate-pulse">Hybrid AI Engine</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-8 text-center max-w-md">
          <AlertCircle className="w-10 h-10 mx-auto mb-3 text-red-500/60" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  const poisson = analysis.poisson;
  const corners = analysis.corners;
  const cards = analysis.cards;

  return (
    <div className="max-w-3xl mx-auto p-6 animate-fade-in">
      {/* ── Match Header ─────────────────────────────── */}
      <div className="bg-surface-2 border border-border rounded-2xl p-6 mb-6">
        <div className="flex items-center justify-center gap-2 mb-5">
          <img src={fixture.league.logo} alt="" className="w-4 h-4" />
          <span className="text-[11px] text-slate-500 uppercase tracking-[0.15em] font-semibold">{fixture.league.name}</span>
          <span className="text-slate-700 mx-1">·</span>
          <span className="text-[11px] text-slate-500 uppercase tracking-[0.15em]">{fixture.time}</span>
        </div>
        <div className="flex items-center justify-center gap-8">
          <div className="text-center w-36">
            <img src={fixture.home_team.logo} alt="" className="w-14 h-14 mx-auto mb-2 drop-shadow-lg" />
            <p className="text-sm font-semibold text-white">{fixture.home_team.name}</p>
          </div>
          <div className="text-2xl font-bold text-slate-600 tracking-widest select-none">VS</div>
          <div className="text-center w-36">
            <img src={fixture.away_team.logo} alt="" className="w-14 h-14 mx-auto mb-2 drop-shadow-lg" />
            <p className="text-sm font-semibold text-white">{fixture.away_team.name}</p>
          </div>
        </div>
      </div>

      {/* ── Poisson Expected Goals (xG) Panel ─────────── */}
      {poisson && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Sigma className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-cyan-300 uppercase">Poisson Expected Goals</h3>
          </div>

          {/* Lambda + Strengths */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-surface-2 border border-cyan-500/15 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">{fixture.home_team.name}</span>
                <span className="text-[10px] text-cyan-400/70 font-bold">HOME</span>
              </div>
              <div className="text-center mb-3">
                <span className="text-3xl font-mono font-black text-cyan-400">{poisson.lambda_home.toFixed(2)}</span>
                <p className="text-[10px] text-slate-500 mt-1">λ expected goals</p>
              </div>
              <div className="flex justify-between text-[10px]">
                <span className="text-slate-500">ATK: <span className={`font-bold ${getStrengthColor(poisson.strengths.home_attack)}`}>{poisson.strengths.home_attack.toFixed(2)}</span></span>
                <span className="text-slate-500">DEF: <span className={`font-bold ${getStrengthColor(2 - poisson.strengths.home_defense)}`}>{poisson.strengths.home_defense.toFixed(2)}</span></span>
              </div>
            </div>
            <div className="bg-surface-2 border border-orange-500/15 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">{fixture.away_team.name}</span>
                <span className="text-[10px] text-orange-400/70 font-bold">AWAY</span>
              </div>
              <div className="text-center mb-3">
                <span className="text-3xl font-mono font-black text-orange-400">{poisson.lambda_away.toFixed(2)}</span>
                <p className="text-[10px] text-slate-500 mt-1">λ expected goals</p>
              </div>
              <div className="flex justify-between text-[10px]">
                <span className="text-slate-500">ATK: <span className={`font-bold ${getStrengthColor(poisson.strengths.away_attack)}`}>{poisson.strengths.away_attack.toFixed(2)}</span></span>
                <span className="text-slate-500">DEF: <span className={`font-bold ${getStrengthColor(2 - poisson.strengths.away_defense)}`}>{poisson.strengths.away_defense.toFixed(2)}</span></span>
              </div>
            </div>
          </div>

          {/* Top Scorelines */}
          {poisson.top_scorelines && poisson.top_scorelines.length > 0 && (
            <div className="bg-surface-2 border border-border rounded-xl p-4 mb-4">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-3">Most Likely Scorelines</p>
              <div className="flex gap-2 justify-center flex-wrap">
                {poisson.top_scorelines.map((s, i) => (
                  <div key={i} className={`px-4 py-2.5 rounded-lg border text-center min-w-[72px] ${i === 0 ? 'bg-cyan-400/10 border-cyan-400/30' : 'bg-surface-1 border-border'}`}>
                    <span className={`text-lg font-mono font-black ${i === 0 ? 'text-cyan-400' : 'text-white'}`}>{s.score}</span>
                    <p className="text-[9px] text-slate-500 mt-0.5">{s.probability}%</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Goals Markets */}
          <div className="bg-surface-2 border border-border rounded-xl p-4 mb-4">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-3">Goals Markets (Poisson)</p>
            <div className="space-y-2">
              {poisson.goals_markets.map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-[12px] text-slate-400 font-medium w-36">{m.market}</span>
                  <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden border border-white/5">
                    <div className={`h-full bg-gradient-to-r ${getBarColor(m.probability)} rounded-full animate-grow`} style={{ width: `${Math.min(m.probability, 100)}%` }} />
                  </div>
                  <span className="text-[12px] font-mono font-bold text-white w-14 text-right">{m.probability.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Result + BTTS */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-surface-2 border border-border rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Result</p>
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-slate-400">{fixture.home_team.name}</span>
                  <span className="text-[12px] font-mono font-bold text-emerald-400">{poisson.result.home_win.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-slate-400">Draw</span>
                  <span className="text-[12px] font-mono font-bold text-yellow-400">{poisson.result.draw.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-slate-400">{fixture.away_team.name}</span>
                  <span className="text-[12px] font-mono font-bold text-red-400">{poisson.result.away_win.toFixed(1)}%</span>
                </div>
              </div>
            </div>
            <div className="bg-surface-2 border border-border rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Both Teams To Score</p>
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-slate-400">Yes</span>
                  <span className="text-[12px] font-mono font-bold text-emerald-400">{poisson.btts.yes.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-slate-400">No</span>
                  <span className="text-[12px] font-mono font-bold text-red-400">{poisson.btts.no.toFixed(1)}%</span>
                </div>
              </div>
              <div className="mt-3 pt-2 border-t border-border">
                <p className="text-[10px] text-slate-500 text-center">
                  Total xG: <span className="text-cyan-400 font-bold">{poisson.expected_total.toFixed(2)}</span>
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Corners Model ────────────────────────────── */}
      {corners && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <CornerUpRight className="w-4 h-4 text-blue-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-blue-300 uppercase">Expected Corners</h3>
            <span className="text-[9px] text-blue-400/50 font-medium ml-auto">
              {fixture.home_team.name}: {corners.expected_home} | {fixture.away_team.name}: {corners.expected_away}
            </span>
          </div>
          <div className="bg-surface-2 border border-blue-500/15 rounded-xl p-4">
            <div className="text-center mb-3">
              <span className="text-2xl font-mono font-black text-blue-400">{corners.expected_total.toFixed(1)}</span>
              <p className="text-[10px] text-slate-500 mt-0.5">Expected Total Corners</p>
            </div>
            <div className="space-y-2">
              {corners.markets.map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-[12px] text-slate-400 font-medium w-36">{m.market}</span>
                  <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden border border-white/5">
                    <div className={`h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full animate-grow`} style={{ width: `${Math.min(m.probability, 100)}%` }} />
                  </div>
                  <span className="text-[12px] font-mono font-bold text-white w-14 text-right">{m.probability.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Cards Model ──────────────────────────────── */}
      {cards && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <CreditCard className="w-4 h-4 text-amber-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-amber-300 uppercase">Expected Cards</h3>
            <span className="text-[9px] text-amber-400/50 font-medium ml-auto">
              {fixture.home_team.name}: {cards.expected_home} | {fixture.away_team.name}: {cards.expected_away}
            </span>
          </div>
          <div className="bg-surface-2 border border-amber-500/15 rounded-xl p-4">
            <div className="text-center mb-3">
              <span className="text-2xl font-mono font-black text-amber-400">{cards.expected_total.toFixed(1)}</span>
              <p className="text-[10px] text-slate-500 mt-0.5">Expected Total Cards</p>
            </div>
            <div className="space-y-2">
              {cards.markets.map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-[12px] text-slate-400 font-medium w-36">{m.market}</span>
                  <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden border border-white/5">
                    <div className={`h-full bg-gradient-to-r from-amber-500 to-amber-400 rounded-full animate-grow`} style={{ width: `${Math.min(m.probability, 100)}%` }} />
                  </div>
                  <span className="text-[12px] font-mono font-bold text-white w-14 text-right">{m.probability.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Value Detections ─────────────────────────── */}
      {analysis.value_selections && analysis.value_selections.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Trophy className="w-4 h-4 text-amber-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-amber-300 uppercase">Value Detections</h3>
          </div>
          <div className="space-y-2">
            {analysis.value_selections.map((vs, idx) => {
              const vstyle = getVerdictStyle(vs.verdict);
              return (
                <div key={idx} className={`bg-surface-2 border rounded-xl p-3 flex items-center justify-between animate-slide-in ${vs.verdict === 'Best Choice' ? 'border-amber-400/30' : 'border-border'}`} style={{ animationDelay: `${idx * 0.06}s` }}>
                  <div className="flex items-center gap-2">
                    <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider border ${vstyle.cls}`}>
                      {vstyle.icon}
                      {vs.verdict}
                    </span>
                    <span className="text-white text-sm font-medium">{vs.pattern}</span>
                    {vs.pillar && (
                      <span className="flex items-center gap-0.5 text-[9px] text-slate-600">
                        {getPillarIcon(vs.pillar)} {vs.pillar}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className="text-slate-500">Prob: <span className="text-white font-bold">{vs.ic}%</span></span>
                    <span className={`font-bold ${vs.value_edge > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      Edge: {vs.value_edge > 0 ? '+' : ''}{vs.value_edge}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── XGBoost Reinforcement ─────────────────────── */}
      {analysis.xgboost_predictions && analysis.xgboost_predictions.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-purple-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-purple-300 uppercase">XGBoost Reinforcement</h3>
          </div>
          <div className="space-y-2">
            {analysis.xgboost_predictions.map((pred, idx) => {
              const badge = getConfBadge(pred.confidence);
              return (
                <div key={idx} className="bg-surface-2 border border-border rounded-xl p-3 hover:border-white/10 transition-all duration-200 animate-slide-in" style={{ animationDelay: `${idx * 0.04}s` }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-[13px] text-white font-medium">{pred.market}</span>
                      <span className="text-purple-400 font-mono font-bold text-[13px]">{pred.probability.toFixed(1)}%</span>
                    </div>
                    <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider border ${badge.cls}`}>
                      {badge.icon}
                      {pred.confidence}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Key Averages ─────────────────────────────── */}
      {analysis.averages && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4 text-slate-400" />
            <h3 className="text-xs font-bold tracking-[0.15em] text-slate-300 uppercase">Key Averages</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-surface-2 border border-border rounded-xl p-4 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-[0.12em] font-bold mb-2">{fixture.home_team.name}</p>
              <div className="flex items-baseline justify-center gap-1">
                <span className="text-2xl font-mono font-bold text-emerald-400">{analysis.averages.home.avg_goals_scored.toFixed(1)}</span>
                <span className="text-slate-600 text-lg">/</span>
                <span className="text-2xl font-mono font-bold text-red-400">{analysis.averages.home.avg_goals_conceded.toFixed(1)}</span>
              </div>
              <p className="text-[10px] text-slate-600 mt-1">scored / conceded</p>
              <div className="flex justify-center gap-3 mt-2 pt-2 border-t border-border">
                <span className="text-[10px] text-blue-400">⛳ {analysis.averages.home.avg_corners}</span>
                <span className="text-[10px] text-amber-400">🟨 {analysis.averages.home.avg_cards}</span>
              </div>
            </div>
            <div className="bg-surface-2 border border-border rounded-xl p-4 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-[0.12em] font-bold mb-2">{fixture.away_team.name}</p>
              <div className="flex items-baseline justify-center gap-1">
                <span className="text-2xl font-mono font-bold text-emerald-400">{analysis.averages.away.avg_goals_scored.toFixed(1)}</span>
                <span className="text-slate-600 text-lg">/</span>
                <span className="text-2xl font-mono font-bold text-red-400">{analysis.averages.away.avg_goals_conceded.toFixed(1)}</span>
              </div>
              <p className="text-[10px] text-slate-600 mt-1">scored / conceded</p>
              <div className="flex justify-center gap-3 mt-2 pt-2 border-t border-border">
                <span className="text-[10px] text-blue-400">⛳ {analysis.averages.away.avg_corners}</span>
                <span className="text-[10px] text-amber-400">🟨 {analysis.averages.away.avg_cards}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Disclaimer ───────────────────────────────── */}
      {analysis.disclaimer && (
        <div className="bg-purple-500/5 border border-purple-500/15 rounded-xl p-4 flex items-start gap-3">
          <Brain className="w-4 h-4 text-purple-400/70 shrink-0 mt-0.5" />
          <p className="text-[11px] text-slate-500 leading-relaxed">
            <strong className="text-purple-400/80">HYBRID ENGINE:</strong> {analysis.disclaimer}
          </p>
        </div>
      )}
    </div>
  );
};

export default MatchDetail;
