import React, { useState, useEffect } from 'react';
import { type Job, type Project, type ProjectCreate, projectApi } from '../../api';
import {
  FolderKanban,
  Plus,
  Play,
  Pause,
  Sliders,
  Sparkles,
  CheckCircle2,
  Music,
  ArrowLeft,
  Trash2,
  Edit2,
  Clock,
  FolderPlus
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface ProjectsViewProps {
  allJobs: Job[];
  onOpenWorkspace: (job: Job) => void;
  onPlay: (job: Job) => void;
  playingSongId?: string;
  isPlayingAudio?: boolean;
  onGenerateInProject: (project: Project) => void;
}

export const ProjectsView: React.FC<ProjectsViewProps> = ({
  allJobs,
  onOpenWorkspace,
  onPlay,
  playingSongId,
  isPlayingAudio,
  onGenerateInProject
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Modals
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isAddTrackModalOpen, setIsAddTrackModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);

  // Form State
  const [newProject, setNewProject] = useState<ProjectCreate>({
    name: '',
    description: '',
    tags: 'Pop, Electronic, Synthwave',
    bpm: 120,
    key_signature: 'C Major',
    color: 'teal'
  });

  const [editProjectData, setEditProjectData] = useState<ProjectCreate>({
    name: '',
    description: '',
    tags: '',
    bpm: 120,
    key_signature: 'C Major',
    color: 'teal'
  });

  const loadProjects = async () => {
    try {
      setIsLoading(true);
      const list = await projectApi.listProjects();
      setProjects(list);
      if (activeProject) {
        const refreshed = await projectApi.getProject(activeProject.id);
        setActiveProject(refreshed);
      }
    } catch (err) {
      console.error('Failed to load projects:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProject.name.trim()) return;

    try {
      const created = await projectApi.createProject(newProject);
      setProjects([created, ...projects]);
      setIsCreateModalOpen(false);
      setNewProject({
        name: '',
        description: '',
        tags: 'Pop, Electronic, Synthwave',
        bpm: 120,
        key_signature: 'C Major',
        color: 'teal'
      });
      // Automatically open the new project
      setActiveProject(created);
    } catch (err) {
      console.error('Failed to create project:', err);
    }
  };

  const handleUpdateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeProject || !editProjectData.name.trim()) return;

    try {
      const updated = await projectApi.updateProject(activeProject.id, editProjectData);
      setActiveProject(updated);
      setProjects(projects.map((p) => (p.id === updated.id ? updated : p)));
      setIsEditModalOpen(false);
    } catch (err) {
      console.error('Failed to update project:', err);
    }
  };

  const handleDeleteProject = async (projectId: string) => {
    if (!confirm('Are you sure you want to delete this project folder? (Sessions will be kept in your general library)')) return;

    try {
      await projectApi.deleteProject(projectId);
      setProjects(projects.filter((p) => p.id !== projectId));
      if (activeProject?.id === projectId) {
        setActiveProject(null);
      }
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  };

  const handleAddTrack = async (jobId: string) => {
    if (!activeProject) return;
    try {
      await projectApi.addTrackToProject(activeProject.id, jobId);
      const updated = await projectApi.getProject(activeProject.id);
      setActiveProject(updated);
      setIsAddTrackModalOpen(false);
      loadProjects();
    } catch (err) {
      console.error('Failed to add track to project:', err);
    }
  };

  const handleRemoveTrack = async (jobId: string) => {
    if (!activeProject) return;
    try {
      await projectApi.removeTrackFromProject(activeProject.id, jobId);
      const updated = await projectApi.getProject(activeProject.id);
      setActiveProject(updated);
      loadProjects();
    } catch (err) {
      console.error('Failed to remove track from project:', err);
    }
  };

  const formatDuration = (seconds: number = 0) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getColorClasses = (color: string = 'teal') => {
    switch (color) {
      case 'cyan':
        return {
          badge: 'bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border-cyan-500/30',
          gradient: 'from-cyan-500 to-blue-600',
          accent: 'text-cyan-500'
        };
      case 'amber':
        return {
          badge: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30',
          gradient: 'from-amber-500 to-orange-600',
          accent: 'text-amber-500'
        };
      case 'emerald':
        return {
          badge: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
          gradient: 'from-emerald-500 to-teal-700',
          accent: 'text-emerald-500'
        };
      case 'sky':
        return {
          badge: 'bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/30',
          gradient: 'from-sky-500 to-indigo-600',
          accent: 'text-sky-500'
        };
      case 'teal':
      default:
        return {
          badge: 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border-teal-500/30',
          gradient: 'from-teal-500 to-cyan-600',
          accent: 'text-teal-500'
        };
    }
  };

  // -------------------------------------------------------------
  // VIEW 1: INSIDE PROJECT FOLDER (Sessions Specific to this Project)
  // -------------------------------------------------------------
  if (activeProject) {
    const projectJobs = allJobs.filter((j) => j.project_id === activeProject.id);
    const colorStyle = getColorClasses(activeProject.color);

    return (
      <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 animate-fade-in">
        {/* Breadcrumb & Top Actions */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <button
            onClick={() => setActiveProject(null)}
            className="inline-flex items-center space-x-2 text-xs font-bold text-slate-500 hover:text-slate-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            <span>All Project Folders</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setEditProjectData({
                  name: activeProject.name,
                  description: activeProject.description || '',
                  tags: activeProject.tags || '',
                  bpm: activeProject.bpm || 120,
                  key_signature: activeProject.key_signature || 'C Major',
                  color: activeProject.color || 'teal'
                });
                setIsEditModalOpen(true);
              }}
              title="Edit project folder details (BPM, Key Signature, Color)"
              aria-label="Edit Project Folder"
              className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 text-xs font-bold transition-colors flex items-center gap-1.5"
            >
              <Edit2 size={13} />
              <span>Edit Project</span>
            </button>

            <button
              onClick={() => handleDeleteProject(activeProject.id)}
              title="Delete this project folder"
              aria-label="Delete Project Folder"
              className="p-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs font-bold transition-colors flex items-center gap-1.5"
            >
              <Trash2 size={13} />
              <span>Delete Folder</span>
            </button>
          </div>
        </div>

        {/* Project Header Banner */}
        <GlassCard className="p-6 md:p-8 space-y-4 border-teal-500/20 relative overflow-hidden">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex items-start space-x-4">
              <div
                className={`w-16 h-16 rounded-2xl bg-gradient-to-tr ${colorStyle.gradient} p-0.5 shadow-apple-md flex-shrink-0 flex items-center justify-center`}
              >
                <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                  <FolderKanban size={28} className="text-white" />
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">
                    {activeProject.name}
                  </h1>
                  <span
                    className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border ${colorStyle.badge}`}
                  >
                    {activeProject.bpm || 120} BPM
                  </span>
                  <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/10 dark:border-white/10">
                    {activeProject.key_signature || 'C Major'}
                  </span>
                </div>

                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">
                  {activeProject.description || 'Dedicated production workspace for multi-session arrangement and stems.'}
                </p>

                {activeProject.tags && (
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {activeProject.tags.split(',').map((t, idx) => (
                      <span
                        key={idx}
                        className="text-[10px] font-mono text-slate-500 dark:text-slate-400 bg-black/[0.03] dark:bg-white/[0.04] px-2 py-0.5 rounded-md"
                      >
                        #{t.trim()}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Quick Actions in Project */}
            <div className="flex flex-wrap items-center gap-2 w-full md:w-auto justify-end">
              <button
                onClick={() => onGenerateInProject(activeProject)}
                title="Open Composer pre-configured with this Project's BPM and Key Signature"
                aria-label="Generate in this Project"
                className="px-4 py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-2 shadow-md shadow-teal-500/20 active:scale-[0.98] transition-transform"
              >
                <Sparkles size={15} />
                <span>Generate in this Project</span>
              </button>

              <button
                onClick={() => setIsAddTrackModalOpen(true)}
                title="Add an existing track from library into this project"
                aria-label="Add Existing Track"
                className="px-3 py-2.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-colors border border-black/[0.06] dark:border-white/10"
              >
                <Plus size={15} />
                <span>Add Existing Track</span>
              </button>
            </div>
          </div>

          {/* Project Stats Meter */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-black/[0.06] dark:border-white/10 text-xs font-mono">
            <div>
              <span className="text-slate-400 text-[10px] block uppercase font-bold">Sessions</span>
              <span className="text-sm font-bold text-slate-900 dark:text-white">
                {projectJobs.length} tracks
              </span>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block uppercase font-bold">Total Duration</span>
              <span className="text-sm font-bold text-slate-900 dark:text-white">
                {formatDuration(
                  projectJobs.reduce((acc, j) => acc + (j.duration_ms || 0) / 1000, 0)
                )}
              </span>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block uppercase font-bold">MIDI Stems</span>
              <span className="text-sm font-bold text-teal-600 dark:text-teal-400">
                {projectJobs.filter((j) => j.midi_path).length} ready
              </span>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block uppercase font-bold">Fast 4-Stems</span>
              <span className="text-sm font-bold text-sky-600 dark:text-sky-400">
                {projectJobs.filter((j) => j.stems_json).length} isolated
              </span>
            </div>
          </div>
        </GlassCard>

        {/* Sessions in this Project */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Project Sessions ({projectJobs.length})
            </h3>
          </div>

          {projectJobs.length === 0 ? (
            <GlassCard className="p-12 text-center space-y-4">
              <div className="w-16 h-16 rounded-3xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center mx-auto">
                <FolderPlus size={32} />
              </div>
              <div className="space-y-1">
                <h4 className="text-base font-bold text-slate-900 dark:text-white">
                  No sessions in this project yet
                </h4>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Click below to generate a new music session directly conditioned for this project's tempo ({activeProject.bpm} BPM) and key ({activeProject.key_signature}).
                </p>
              </div>
              <div className="flex justify-center gap-3 pt-2">
                <button
                  onClick={() => onGenerateInProject(activeProject)}
                  className="px-5 py-2.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl shadow-md flex items-center space-x-2"
                >
                  <Sparkles size={15} />
                  <span>Generate First Session</span>
                </button>
                <button
                  onClick={() => setIsAddTrackModalOpen(true)}
                  className="px-4 py-2.5 bg-black/[0.04] dark:bg-white/5 text-slate-700 dark:text-slate-300 font-bold text-xs rounded-xl"
                >
                  Add Existing Track
                </button>
              </div>
            </GlassCard>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {projectJobs.map((job) => {
                const isThisPlaying = isPlayingAudio && playingSongId === job.id;

                return (
                  <GlassCard
                    key={job.id}
                    className="p-5 space-y-4 hover:border-teal-500/30 transition-all flex flex-col justify-between group"
                  >
                    <div className="space-y-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                            {job.title || job.prompt.slice(0, 30)}
                          </h4>
                          <p className="text-[11px] font-mono text-teal-600 dark:text-teal-400 truncate mt-0.5">
                            {job.tags || activeProject.tags || 'Studio Mix'}
                          </p>
                        </div>

                        <span
                          className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${
                            job.status === 'completed'
                              ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20'
                              : 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20'
                          }`}
                        >
                          {job.status.toUpperCase()}
                        </span>
                      </div>

                      {/* Capabilities Checklist */}
                      <div className="space-y-1.5 pt-1">
                        <div className="flex items-center justify-between text-[11px] font-mono">
                          <span className="text-slate-500 dark:text-slate-400 flex items-center gap-1">
                            <CheckCircle2 size={12} className="text-teal-500" />
                            AI Multi-Track Score
                          </span>
                          <span className="text-teal-600 dark:text-teal-400 font-bold">
                            {job.midi_path ? 'MIDI + Score' : 'Processing'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[11px] font-mono">
                          <span className="text-slate-500 dark:text-slate-400 flex items-center gap-1">
                            <CheckCircle2 size={12} className="text-sky-500" />
                            Fast 4-Stems
                          </span>
                          <span className="text-sky-600 dark:text-sky-400 font-bold">
                            {job.stems_json ? 'Vocals/Drums/Bass/Inst' : 'Ready'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Action Bar */}
                    <div className="flex items-center justify-between pt-3 border-t border-black/[0.06] dark:border-white/10 gap-2">
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => onPlay(job)}
                          className="p-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold transition-transform active:scale-95 shadow-sm"
                          title={isThisPlaying ? 'Pause' : 'Play Track Preview'}
                        >
                          {isThisPlaying ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                        </button>
                        <span className="text-[10px] font-mono text-slate-400">
                          {formatDuration((job.duration_ms || 0) / 1000)}
                        </span>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => onOpenWorkspace(job)}
                          className="px-3 py-1.5 bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold text-xs rounded-xl transition-all flex items-center gap-1"
                        >
                          <Sliders size={13} />
                          <span>DAW</span>
                        </button>

                        <button
                          onClick={() => handleRemoveTrack(job.id)}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                          title="Remove from Project"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  </GlassCard>
                );
              })}
            </div>
          )}
        </div>

        {/* Modal: Add Existing Track to Project */}
        {isAddTrackModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="bg-white dark:bg-[#181a24] rounded-3xl border border-black/[0.08] dark:border-white/10 p-6 max-w-md w-full shadow-apple-lg space-y-4">
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Add Track to {activeProject.name}
              </h3>
              <p className="text-xs text-slate-500">
                Choose a completed generation from your library to associate with this project folder.
              </p>

              <div className="max-h-60 overflow-y-auto space-y-1.5">
                {allJobs
                  .filter((j) => j.status === 'completed' && j.project_id !== activeProject.id)
                  .map((song) => (
                    <div
                      key={song.id}
                      onClick={() => handleAddTrack(song.id)}
                      className="p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] hover:bg-teal-500/10 border border-transparent hover:border-teal-500/30 cursor-pointer flex items-center justify-between transition-all"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate">
                          {song.title || song.prompt.slice(0, 35)}
                        </div>
                        <div className="text-[10px] font-mono text-slate-400 truncate">
                          {song.tags || 'Music'}
                        </div>
                      </div>
                      <Plus size={14} className="text-teal-500 flex-shrink-0 ml-2" />
                    </div>
                  ))}
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setIsAddTrackModalOpen(false)}
                  className="px-4 py-2 rounded-xl bg-black/5 dark:bg-white/5 text-slate-700 dark:text-slate-300 font-bold text-xs"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Modal: Edit Project */}
        {isEditModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="bg-white dark:bg-[#181a24] rounded-3xl border border-black/[0.08] dark:border-white/10 p-6 max-w-md w-full shadow-apple-lg space-y-4">
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Edit Project Folder
              </h3>

              <form onSubmit={handleUpdateProject} className="space-y-3">
                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Project Name
                  </label>
                  <input
                    type="text"
                    required
                    value={editProjectData.name}
                    onChange={(e) => setEditProjectData({ ...editProjectData, name: e.target.value })}
                    className="apple-input text-xs"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Description
                  </label>
                  <textarea
                    rows={2}
                    value={editProjectData.description}
                    onChange={(e) => setEditProjectData({ ...editProjectData, description: e.target.value })}
                    className="apple-input text-xs"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                      Tempo (BPM)
                    </label>
                    <input
                      type="number"
                      min={40}
                      max={240}
                      value={editProjectData.bpm || 120}
                      onChange={(e) => setEditProjectData({ ...editProjectData, bpm: parseInt(e.target.value) })}
                      className="apple-input text-xs"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                      Musical Key
                    </label>
                    <input
                      type="text"
                      value={editProjectData.key_signature || 'C Major'}
                      onChange={(e) => setEditProjectData({ ...editProjectData, key_signature: e.target.value })}
                      className="apple-input text-xs"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Default Style / Tags
                  </label>
                  <input
                    type="text"
                    value={editProjectData.tags}
                    onChange={(e) => setEditProjectData({ ...editProjectData, tags: e.target.value })}
                    className="apple-input text-xs"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Accent Color
                  </label>
                  <div className="flex gap-2">
                    {['teal', 'cyan', 'amber', 'emerald', 'sky'].map((c) => (
                      <button
                        key={c}
                        type="button"
                        onClick={() => setEditProjectData({ ...editProjectData, color: c })}
                        className={`w-7 h-7 rounded-full border-2 transition-transform ${
                          c === 'teal'
                            ? 'bg-teal-500'
                            : c === 'cyan'
                            ? 'bg-cyan-500'
                            : c === 'amber'
                            ? 'bg-amber-500'
                            : c === 'emerald'
                            ? 'bg-emerald-500'
                            : 'bg-sky-500'
                        } ${
                          editProjectData.color === c
                            ? 'border-white scale-110 shadow-md'
                            : 'border-transparent opacity-60'
                        }`}
                      />
                    ))}
                  </div>
                </div>

                <div className="flex justify-end gap-2 pt-3">
                  <button
                    type="button"
                    onClick={() => setIsEditModalOpen(false)}
                    className="px-4 py-2 rounded-xl bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-300 font-bold text-xs"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-5 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-sm"
                  >
                    Save Changes
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    );
  }

  // -------------------------------------------------------------
  // VIEW 2: TOP LEVEL PROJECTS FOLDER BROWSER
  // -------------------------------------------------------------
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
            <span className="p-2 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
              📁
            </span>
            <span>Studio Project Folders</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
            Organize multi-session recordings, arranged tracks, and transcribed stems by project
          </p>
        </div>

        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="px-4 py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-2xl shadow-apple-md flex items-center space-x-2 transition-all hover:scale-105 active:scale-95 flex-shrink-0"
        >
          <FolderPlus size={16} />
          <span>New Project Folder +</span>
        </button>
      </div>

      {/* Projects Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((n) => (
            <div key={n} className="h-48 rounded-3xl bg-black/5 dark:bg-white/5 animate-pulse" />
          ))}
        </div>
      ) : projects.length === 0 ? (
        <GlassCard className="p-12 text-center space-y-4">
          <div className="w-16 h-16 rounded-3xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center mx-auto">
            <FolderKanban size={32} />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              No Studio Projects Yet
            </h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Create your first project folder to keep related generation sessions, multitrack stems, and MIDI arrangements organized.
            </p>
          </div>
          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="px-5 py-2.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl shadow-md inline-flex items-center space-x-2"
          >
            <FolderPlus size={16} />
            <span>Create First Project</span>
          </button>
        </GlassCard>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => {
            const colorStyle = getColorClasses(project.color);
            const projectJobs = allJobs.filter((j) => j.project_id === project.id);
            const totalDuration = projectJobs.reduce((acc, j) => acc + (j.duration_ms || 0) / 1000, 0);

            return (
              <GlassCard
                key={project.id}
                onClick={() => setActiveProject(project)}
                className="p-6 space-y-4 hover:border-teal-500/40 cursor-pointer transition-all hover:scale-[1.01] group flex flex-col justify-between"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div
                      className={`w-12 h-12 rounded-2xl bg-gradient-to-tr ${colorStyle.gradient} p-0.5 shadow-md flex items-center justify-center flex-shrink-0`}
                    >
                      <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                        <FolderKanban size={22} className="text-white" />
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5 flex-wrap justify-end">
                      <span
                        className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${colorStyle.badge}`}
                      >
                        {project.bpm || 120} BPM
                      </span>
                      <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/10 dark:border-white/10">
                        {project.key_signature || 'C Major'}
                      </span>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                      {project.name}
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2 mt-1">
                      {project.description || 'Studio sessions container'}
                    </p>
                  </div>
                </div>

                <div className="pt-3 border-t border-black/[0.06] dark:border-white/10 flex items-center justify-between text-xs font-mono text-slate-500 dark:text-slate-400">
                  <span className="flex items-center gap-1">
                    <Music size={12} className="text-teal-500" />
                    {projectJobs.length} {projectJobs.length === 1 ? 'session' : 'sessions'}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock size={12} />
                    {formatDuration(totalDuration)}
                  </span>
                </div>
              </GlassCard>
            );
          })}
        </div>
      )}

      {/* Modal: Create New Project */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-[#181a24] rounded-3xl border border-black/[0.08] dark:border-white/10 p-6 max-w-md w-full shadow-apple-lg space-y-4">
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <FolderPlus size={18} className="text-teal-500" />
              <span>Create New Project Folder</span>
            </h3>

            <form onSubmit={handleCreateProject} className="space-y-3">
              <div>
                <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                  Project Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g., Midnight Neon Album"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  className="apple-input text-xs"
                />
              </div>

              <div>
                <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                  Description
                </label>
                <textarea
                  rows={2}
                  placeholder="What is the concept or sound goal of this project?"
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                  className="apple-input text-xs"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Tempo (BPM)
                  </label>
                  <input
                    type="number"
                    min={40}
                    max={240}
                    value={newProject.bpm}
                    onChange={(e) => setNewProject({ ...newProject, bpm: parseInt(e.target.value) })}
                    className="apple-input text-xs"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                    Musical Key
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., D Minor"
                    value={newProject.key_signature}
                    onChange={(e) => setNewProject({ ...newProject, key_signature: e.target.value })}
                    className="apple-input text-xs"
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                  Default Style Tags
                </label>
                <input
                  type="text"
                  placeholder="e.g., Synthwave, Analog, 80s"
                  value={newProject.tags}
                  onChange={(e) => setNewProject({ ...newProject, tags: e.target.value })}
                  className="apple-input text-xs"
                />
              </div>

              <div>
                <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">
                  Folder Accent Color
                </label>
                <div className="flex gap-2">
                  {['teal', 'cyan', 'amber', 'emerald', 'sky'].map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setNewProject({ ...newProject, color: c })}
                      className={`w-7 h-7 rounded-full border-2 transition-transform ${
                        c === 'teal'
                          ? 'bg-teal-500'
                          : c === 'cyan'
                          ? 'bg-cyan-500'
                          : c === 'amber'
                          ? 'bg-amber-500'
                          : c === 'emerald'
                          ? 'bg-emerald-500'
                          : 'bg-sky-500'
                      } ${
                        newProject.color === c
                          ? 'border-white scale-110 shadow-md'
                          : 'border-transparent opacity-60'
                      }`}
                    />
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3">
                <button
                  type="button"
                  onClick={() => setIsCreateModalOpen(false)}
                  className="px-4 py-2 rounded-xl bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-300 font-bold text-xs"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold text-xs shadow-md"
                >
                  Create Project Folder
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
