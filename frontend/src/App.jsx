import React, { useState, useEffect, useCallback } from 'react';
import { Trophy, Loader2, AlertCircle, Calendar, Target } from 'lucide-react';
import MatchRow from './components/MatchRow';
import MatchDetail from './components/MatchDetail';
import DatePicker from './components/DatePicker';
import ResultsTracker from './components/ResultsTracker';

const API = "http://127.0.0.1:8000/api";

const pad = (n) => String(n).padStart(2, '0');
const fmtDate = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function App() {
  const [selectedDate, setSelectedDate] = useState(fmtDate(new Date()));
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('predictions'); // 'predictions' | 'results'

  const [selectedFixtureId, setSelectedFixtureId] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  // Fetch fixtures for the selected date
  const fetchFixtures = useCallback(async (dateStr) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/fixtures/${dateStr}`);
      if (!res.ok) throw new Error("Failed to fetch matches");
      const data = await res.json();
      setFixtures(data);
      // Auto-select first match if none selected
      if (data.length > 0) {
        handleMatchSelect(data[0]);
      } else {
        setSelectedFixtureId(null);
        setAnalysis(null);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchFixtures(selectedDate); }, [selectedDate, fetchFixtures]);

  const handleMatchSelect = async (fixture) => {
    setSelectedFixtureId(fixture.id);
    setAnalysis(null);
    setAnalysisError(null);
    setAnalysisLoading(true);
    try {
      const res = await fetch(`${API}/analysis/match/${fixture.id}?home=${encodeURIComponent(fixture.home_team.name)}&away=${encodeURIComponent(fixture.away_team.name)}&league=${encodeURIComponent(fixture.league.name)}&live_home=${fixture.home_goals || 0}&live_away=${fixture.away_goals || 0}&status=${encodeURIComponent(fixture.status)}&start_time=${encodeURIComponent(fixture.time)}`);
      if (!res.ok) throw new Error("Analysis failed");
      const data = await res.json();
      // Merge fixture display info
      data.match = {
        ...data.match,
        home_team_logo: fixture.home_team.logo,
        away_team_logo: fixture.away_team.logo,
        home_team: fixture.home_team.name,
        away_team: fixture.away_team.name,
        league_name: fixture.league.name,
        league_logo: fixture.league.logo,
        time: fixture.time,
      };
      setAnalysis(data);
    } catch (e) {
      setAnalysisError(e.message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Group fixtures by league
  const grouped = fixtures.reduce((acc, f) => {
    const key = f.league.name;
    if (!acc[key]) acc[key] = { league: f.league, matches: [] };
    acc[key].matches.push(f);
    return acc;
  }, {});

  const selectedFixture = fixtures.find(f => f.id === selectedFixtureId);

  // If in results mode, show the full-screen results tracker
  if (viewMode === 'results') {
    return (
      <div className="h-screen flex flex-col overflow-hidden">
        <header className="shrink-0 h-14 bg-surface-1 border-b border-border flex items-center px-5 gap-3 z-30">
          <Trophy className="w-5 h-5 text-gold-500" />
          <span className="text-base font-bold tracking-widest text-white">
            FOOTBALL<span className="text-gold-500">PREDICT</span>
          </span>
          <div className="flex-1" />
          <button
            onClick={() => setViewMode('predictions')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:border-white/20 transition-all text-xs"
          >
            <Calendar className="w-3.5 h-3.5" />
            Predictions
          </button>
          <button
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-400 text-xs font-bold"
          >
            <Target className="w-3.5 h-3.5" />
            Results
          </button>
        </header>
        <div className="flex-1 overflow-hidden bg-surface-0">
          <ResultsTracker
            onBack={() => setViewMode('predictions')}
            selectedDate={selectedDate}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* ── Top Bar ─────────────────────────────────────── */}
      <header className="shrink-0 h-14 bg-surface-1 border-b border-border flex items-center px-5 gap-3 z-30">
        <Trophy className="w-5 h-5 text-gold-500" />
        <span className="text-base font-bold tracking-widest text-white">
          FOOTBALL<span className="text-gold-500">PREDICT</span>
        </span>
        <div className="flex-1" />
        <button
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gold-500/15 border border-gold-500/30 text-gold-500 text-xs font-bold"
        >
          <Calendar className="w-3.5 h-3.5" />
          Predictions
        </button>
        <button
          onClick={() => setViewMode('results')}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-amber-400 hover:border-amber-500/30 transition-all text-xs"
        >
          <Target className="w-3.5 h-3.5" />
          Results
        </button>
      </header>

      {/* ── Date Picker ──────────────────────────────────── */}
      <DatePicker selectedDate={selectedDate} onDateChange={setSelectedDate} />

      {/* ── Main 2-Panel Layout ──────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel — Match List */}
        <aside className="w-[380px] shrink-0 bg-surface-1 border-r border-border overflow-y-auto">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 text-slate-500">
              <Loader2 className="w-6 h-6 animate-spin text-gold-500 mb-3" />
              <span className="text-xs tracking-widest uppercase">Loading Matches</span>
            </div>
          ) : error ? (
            <div className="p-6 text-center text-red-400 text-sm">
              <AlertCircle className="w-6 h-6 mx-auto mb-2 opacity-50" />
              {error}
            </div>
          ) : fixtures.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-sm">
              No matches found for this date.
            </div>
          ) : (
            Object.values(grouped).map((group, gi) => (
              <div key={group.league.id} className="animate-fade-in" style={{ animationDelay: `${gi * 0.05}s` }}>
                {/* League Header */}
                <div className="sticky top-0 z-10 bg-surface-1/95 backdrop-blur-sm flex items-center gap-2.5 px-4 py-2 border-b border-border">
                  <img src={group.league.logo} alt="" className="w-4 h-4 object-contain" />
                  <span className="text-xs font-semibold tracking-wider text-slate-300 uppercase">{group.league.name}</span>
                  <span className="text-[10px] text-slate-600 ml-auto uppercase">{group.league.country}</span>
                </div>
                {/* Match Rows */}
                {group.matches.map(match => (
                  <MatchRow
                    key={match.id}
                    match={match}
                    isSelected={match.id === selectedFixtureId}
                    onClick={() => handleMatchSelect(match)}
                  />
                ))}
              </div>
            ))
          )}
        </aside>

        {/* Center Panel — Match Detail / Analysis */}
        <main className="flex-1 overflow-y-auto bg-surface-0">
          {selectedFixture ? (
            <MatchDetail
              fixture={selectedFixture}
              analysis={analysis}
              loading={analysisLoading}
              error={analysisError}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-600">
              <Trophy className="w-12 h-12 mb-4 opacity-30" />
              <p className="text-sm">Select a match to view analysis</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
