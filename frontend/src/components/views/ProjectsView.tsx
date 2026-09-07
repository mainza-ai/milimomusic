import React, { useState, useEffect, useRef } from 'react';
import { type Job, type Project, type ProjectCreate, projectApi, coverApi, API_BASE_URL } from '../../api';
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
  FolderPlus,
  Upload,
  Image as ImageIcon,
  X,
  Copy,
  Search,
  Package
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { AppFooter } from '../ui/AppFooter';

interface ProjectsViewProps {
  allJobs: Job[];
  onOpenWorkspace: (job: Job) => void;
  onPlay: (job: Job) => void;
  playingSongId?: string;
  isPlayingAudio?: boolean;
  onGenerateInProject: (project: Project) => void;
  onSelectTrack?: (job: Job) => void;
}

export const ProjectsView: React.FC<ProjectsViewProps> = ({
  allJobs,
  onOpenWorkspace,
  onPlay,
  playingSongId,
  isPlayingAudio,
  onGenerateInProject,
  onSelectTrack
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Modals
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isAddTrackModalOpen, setIsAddTrackModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);

  // Image Upload / AI Cover State
  const [isUploadingCover, setIsUploadingCover] = useState(false);
  const [isGeneratingCover, setIsGeneratingCover] = useState(false);
  const [coverTarget, setCoverTarget] = useState<'create' | 'edit'>('create');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Search & Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState('all');

  // Form State
  const [newProject, setNewProject] = useState<ProjectCreate>({
    name: '',
    description: '',
    cover_image_path: '',
    image_prompt: '',
    tags: 'Pop, Electronic, Synthwave',
    bpm: 120,
    key_signature: 'C Major',
    color: 'teal'
  });

  const [editProjectData, setEditProjectData] = useState<ProjectCreate>({
    name: '',
    description: '',
    cover_image_path: '',
    image_prompt: '',
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

  const handleFileUpload = async (file: File) => {
    try {
      setIsUploadingCover(true);
      const res = await coverApi.uploadCoverImage(file);
      const fullUrl = res.url.startsWith('http') ? res.url : `${API_BASE_URL}${res.url}`;
      if (coverTarget === 'create') {
        setNewProject(prev => ({ ...prev, cover_image_path: fullUrl }));
      } else {
        setEditProjectData(prev => ({ ...prev, cover_image_path: fullUrl }));
      }
    } catch (err) {
      console.error('Failed to upload image:', err);
    } finally {
      setIsUploadingCover(false);
    }
  };

  const handlePromptCover = async () => {
    try {
      setIsGeneratingCover(true);
      const targetData = coverTarget === 'create' ? newProject : editProjectData;
      const promptRes = await coverApi.generateCoverPrompt({
        title: targetData.name || 'Studio Project',
        description: targetData.description,
        tags: targetData.tags
      });
      const imgRes = await coverApi.generateCoverImage({ prompt: promptRes.prompt });
      const fullUrl = imgRes.url.startsWith('http') ? imgRes.url : `${API_BASE_URL}${imgRes.url}`;
      if (coverTarget === 'create') {
        setNewProject(prev => ({
          ...prev,
          cover_image_path: fullUrl,
          image_prompt: promptRes.prompt
        }));
      } else {
        setEditProjectData(prev => ({
          ...prev,
          cover_image_path: fullUrl,
          image_prompt: promptRes.prompt
        }));
      }
    } catch (err) {
      console.error('Failed to generate cover:', err);
    } finally {
      setIsGeneratingCover(false);
    }
  };

  const handleDuplicateProject = async (projectId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    try {
      const duplicated = await projectApi.duplicateProject(projectId);
      setProjects(prev => [duplicated, ...prev]);
      if (activeProject?.id === projectId) {
        setActiveProject(duplicated);
      }
    } catch (err) {
      console.error('Failed to duplicate project:', err);
    }
  };

  const handleExportProjectPack = (projectId: string) => {
    window.open(projectApi.exportProjectPackUrl(projectId), '_blank');
  };

  const handlePlayAll = () => {
    if (!activeProject) return;
    const projectJobs = allJobs.filter((j) => j.project_id === activeProject.id);
    if (projectJobs.length > 0) {
      onPlay(projectJobs[0]);
    }
  };

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
        cover_image_path: '',
        image_prompt: '',
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
              onClick={() => handleExportProjectPack(activeProject.id)}
              title="Download full project studio pack (multitrack stems, audio, MIDI, score, lyrics zip)"
              aria-label="Export Project Studio Pack"
              className="p-2 rounded-xl bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 text-xs font-bold transition-colors flex items-center gap-1.5 border border-teal-500/20"
            >
              <Package size={13} />
              <span>Export Studio Pack</span>
            </button>

            <button
              onClick={(e) => handleDuplicateProject(activeProject.id, e)}
              title="Duplicate this project and its settings"
              aria-label="Duplicate Project Folder"
              className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 text-xs font-bold transition-colors flex items-center gap-1.5"
            >
              <Copy size={13} />
              <span>Duplicate</span>
            </button>

            <button
              onClick={() => {
                setEditProjectData({
                  name: activeProject.name,
                  description: activeProject.description || '',
                  cover_image_path: activeProject.cover_image_path || '',
                  image_prompt: activeProject.image_prompt || '',
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
                className={`w-16 h-16 rounded-2xl overflow-hidden bg-slate-900 border border-black/10 dark:border-white/10 shadow-apple-md flex-shrink-0 flex items-center justify-center`}
              >
                {activeProject.cover_image_path ? (
                  <img
                    src={activeProject.cover_image_path.startsWith('http') ? activeProject.cover_image_path : `${API_BASE_URL}${activeProject.cover_image_path}`}
                    alt={activeProject.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className={`w-full h-full bg-gradient-to-tr ${colorStyle.gradient} flex items-center justify-center`}>
                    <FolderKanban size={28} className="text-white" />
                  </div>
                )}
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
              {projectJobs.length > 0 && (
                <button
                  onClick={handlePlayAll}
                  title="Play all tracks in this project starting from the first session"
                  aria-label="Play All Tracks"
                  className="px-3.5 py-2.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-colors border border-black/[0.06] dark:border-white/10"
                >
                  <Play size={14} className="text-teal-500 fill-teal-500" />
                  <span>Play All</span>
                </button>
              )}

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
                        <div
                          onClick={() => onSelectTrack?.(job)}
                          className="min-w-0 cursor-pointer group/title"
                          title="Open Track Studio"
                        >
                          <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 truncate group-hover/title:text-teal-600 dark:group-hover/title:text-teal-400 transition-colors">
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
                    <div className="flex items-center justify-between pt-3 border-t border-black/[0.06] dark:border-white/10 gap-2 flex-wrap">
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
                        {onSelectTrack && (
                          <button
                            onClick={() => onSelectTrack(job)}
                            className="px-2.5 py-1.5 bg-black/5 dark:bg-white/5 hover:bg-teal-500/20 text-slate-700 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 font-bold text-xs rounded-xl transition-all flex items-center gap-1 border border-black/5 dark:border-white/5 shadow-sm"
                            title="Open Track Studio (Stems, MIDI, Score, Provenance)"
                          >
                            <Sparkles size={13} className="text-teal-500" />
                            <span>Studio</span>
                          </button>
                        )}

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
                    Project Artwork
                  </label>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl overflow-hidden bg-slate-900 border border-black/10 dark:border-white/10 flex-shrink-0 relative">
                      {editProjectData.cover_image_path ? (
                        <img
                          src={editProjectData.cover_image_path.startsWith('http') ? editProjectData.cover_image_path : `${API_BASE_URL}${editProjectData.cover_image_path}`}
                          alt="Cover"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-teal-500/20 text-teal-400">
                          <ImageIcon size={18} />
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setCoverTarget('edit');
                          fileInputRef.current?.click();
                        }}
                        disabled={isUploadingCover}
                        className="py-1.5 px-3 rounded-xl bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1 border border-black/10 dark:border-white/10"
                      >
                        <Upload size={12} />
                        <span>{isUploadingCover && coverTarget === 'edit' ? 'Uploading...' : 'Upload'}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setCoverTarget('edit');
                          handlePromptCover();
                        }}
                        disabled={isGeneratingCover}
                        className="py-1.5 px-3 rounded-xl bg-teal-500/10 hover:bg-teal-500/20 text-xs font-semibold text-teal-600 dark:text-teal-400 flex items-center space-x-1 border border-teal-500/20"
                      >
                        <Sparkles size={12} />
                        <span>{isGeneratingCover && coverTarget === 'edit' ? 'Generating...' : 'AI Cover'}</span>
                      </button>
                    </div>
                  </div>
                </div>

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
          className="px-4 py-2.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-2xl shadow-apple-md flex items-center space-x-2 transition-all hover:scale-105 active:scale-95 flex-shrink-0"
        >
          <Plus size={16} />
          <span>New project</span>
        </button>
      </div>

      {/* Search & Tag Filter Bar */}
      {projects.length > 0 && (
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
            <input
              type="text"
              placeholder="Search projects by name, tags, or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-8 py-2 rounded-xl bg-black/[0.03] dark:bg-white/5 border border-black/[0.08] dark:border-white/10 text-xs text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:border-teal-500 transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-white"
              >
                <X size={13} />
              </button>
            )}
          </div>

          {Array.from(new Set(projects.flatMap(p => (p.tags ? p.tags.split(',').map(t => t.trim()) : [])).filter(Boolean))).length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0 scrollbar-none">
              <button
                onClick={() => setSelectedTag('all')}
                className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                  selectedTag === 'all'
                    ? 'bg-teal-500 text-slate-950 shadow-sm'
                    : 'bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                All
              </button>
              {Array.from(new Set(projects.flatMap(p => (p.tags ? p.tags.split(',').map(t => t.trim()) : [])).filter(Boolean)))
                .slice(0, 6)
                .map((tag) => (
                  <button
                    key={tag}
                    onClick={() => setSelectedTag(selectedTag === tag ? 'all' : tag)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                      selectedTag === tag
                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                        : 'bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    #{tag}
                  </button>
                ))}
            </div>
          )}
        </div>
      )}

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
        (() => {
          const q = searchQuery.toLowerCase().trim();
          const filteredProjects = projects.filter((p) => {
            const matchesSearch =
              !q ||
              (p.name || '').toLowerCase().includes(q) ||
              (p.description || '').toLowerCase().includes(q) ||
              (p.tags || '').toLowerCase().includes(q);
            const matchesTag =
              selectedTag === 'all' ||
              (p.tags && p.tags.toLowerCase().includes(selectedTag.toLowerCase()));
            return matchesSearch && matchesTag;
          });

          if (filteredProjects.length === 0) {
            return (
              <GlassCard className="p-8 text-center space-y-3">
                <p className="text-sm font-semibold text-slate-500">
                  No projects matching "{searchQuery}" {selectedTag !== 'all' ? `with tag #${selectedTag}` : ''}
                </p>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedTag('all');
                  }}
                  className="px-4 py-1.5 rounded-xl bg-teal-500/10 hover:bg-teal-500/20 text-teal-600 dark:text-teal-400 text-xs font-bold transition-colors"
                >
                  Reset filters
                </button>
              </GlassCard>
            );
          }

          return (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredProjects.map((project) => {
                const colorStyle = getColorClasses(project.color);
                const projectJobs = allJobs.filter((j) => j.project_id === project.id);
                const totalDuration = projectJobs.reduce((acc, j) => acc + (j.duration_ms || 0) / 1000, 0);

                return (
                  <GlassCard
                    key={project.id}
                    onClick={() => setActiveProject(project)}
                    className="p-5 space-y-4 hover:border-teal-500/40 cursor-pointer transition-all hover:scale-[1.01] group flex flex-col justify-between overflow-hidden"
                  >
                    <div className="space-y-3">
                      <div className="flex items-start gap-3">
                        {/* Project Artwork Thumbnail */}
                        <div className="w-16 h-16 rounded-2xl overflow-hidden bg-slate-900 border border-black/10 dark:border-white/10 flex-shrink-0 shadow-md relative group-hover:scale-105 transition-transform">
                          {project.cover_image_path ? (
                            <img
                              src={project.cover_image_path.startsWith('http') ? project.cover_image_path : `${API_BASE_URL}${project.cover_image_path}`}
                              alt={project.name}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                (e.target as HTMLElement).style.display = 'none';
                              }}
                            />
                          ) : (
                            <div className={`w-full h-full bg-gradient-to-tr ${colorStyle.gradient} flex items-center justify-center`}>
                              <FolderKanban size={24} className="text-white drop-shadow" />
                            </div>
                          )}
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5 flex-wrap justify-end mb-1">
                            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${colorStyle.badge}`}>
                              {project.bpm || 120} BPM
                            </span>
                            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/10 dark:border-white/10">
                              {project.key_signature || 'C Major'}
                            </span>
                          </div>
                          <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors truncate">
                            {project.name}
                          </h3>
                          <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">
                            {project.description || 'Studio sessions container'}
                          </p>
                          {project.tags && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {project.tags.split(',').slice(0, 3).map((t, idx) => (
                                <span
                                  key={idx}
                                  className="text-[9px] font-mono text-slate-500 dark:text-slate-400 bg-black/[0.03] dark:bg-white/[0.04] px-1.5 py-0.5 rounded"
                                >
                                  #{t.trim()}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-black/[0.06] dark:border-white/10 flex items-center justify-between text-xs font-mono text-slate-500 dark:text-slate-400">
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1">
                          <Music size={12} className="text-teal-500" />
                          {projectJobs.length} {projectJobs.length === 1 ? 'song' : 'songs'}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock size={12} />
                          {formatDuration(totalDuration)}
                        </span>
                      </div>

                      <div className="flex items-center gap-1 opacity-70 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleExportProjectPack(project.id);
                          }}
                          title="Export Project Studio Pack (.zip)"
                          className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-500 transition-colors"
                        >
                          <Package size={13} />
                        </button>
                        <button
                          onClick={(e) => handleDuplicateProject(project.id, e)}
                          title="Duplicate Project"
                          className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-500 transition-colors"
                        >
                          <Copy size={13} />
                        </button>
                      </div>
                    </div>
                  </GlassCard>
                );
              })}
            </div>
          );
        })()
      )}

      {/* Hidden File Input for Image Upload */}
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileUpload(file);
        }}
        className="hidden"
      />

      {/* 2-Column Modal: Create New Project (Production Design) */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fade-in">
          <div className="bg-[#181a24] text-white rounded-3xl border border-white/10 p-6 max-w-xl w-full shadow-2xl space-y-5 relative">
            {/* Modal Header */}
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-white">
                New project
              </h3>
              <button
                onClick={() => setIsCreateModalOpen(false)}
                className="p-1.5 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                title="Close"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleCreateProject} className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 items-start">
                {/* Left Column: Drag & Drop Dropzone + Upload & Prompt */}
                <div className="space-y-3">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      const file = e.dataTransfer.files?.[0];
                      if (file) handleFileUpload(file);
                    }}
                    className="aspect-square w-full rounded-2xl bg-[#222533] border-2 border-dashed border-white/15 hover:border-teal-500/50 flex flex-col items-center justify-center text-center p-4 cursor-pointer transition-all relative overflow-hidden group"
                  >
                    {newProject.cover_image_path ? (
                      <>
                        <img
                          src={newProject.cover_image_path.startsWith('http') ? newProject.cover_image_path : `${API_BASE_URL}${newProject.cover_image_path}`}
                          alt="Project Cover"
                          className="w-full h-full object-cover absolute inset-0"
                        />
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <span className="text-xs font-semibold text-white">Change Image</span>
                        </div>
                      </>
                    ) : isUploadingCover || isGeneratingCover ? (
                      <div className="space-y-2">
                        <div className="w-8 h-8 rounded-full border-2 border-teal-400 border-t-transparent animate-spin mx-auto" />
                        <span className="text-xs text-slate-400 font-medium">
                          {isGeneratingCover ? 'Generating artwork...' : 'Uploading image...'}
                        </span>
                      </div>
                    ) : (
                      <div className="space-y-2 text-slate-400">
                        <ImageIcon size={28} className="mx-auto opacity-50 text-slate-400" />
                        <p className="text-xs font-medium px-2 leading-relaxed">
                          Drag and drop project image
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setCoverTarget('create');
                        fileInputRef.current?.click();
                      }}
                      disabled={isUploadingCover}
                      className="py-2 px-3 rounded-xl bg-white/5 hover:bg-white/10 text-xs font-semibold text-slate-300 hover:text-white flex items-center justify-center space-x-1.5 transition-colors border border-white/5"
                    >
                      <Upload size={14} />
                      <span>{isUploadingCover && coverTarget === 'create' ? 'Uploading...' : 'Upload'}</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setCoverTarget('create');
                        handlePromptCover();
                      }}
                      disabled={isGeneratingCover}
                      className="py-2 px-3 rounded-xl bg-white/5 hover:bg-white/10 text-xs font-semibold text-teal-300 hover:text-teal-200 flex items-center justify-center space-x-1.5 transition-colors border border-white/5"
                    >
                      <Sparkles size={14} />
                      <span>{isGeneratingCover && coverTarget === 'create' ? 'Generating...' : 'Prompt'}</span>
                    </button>
                  </div>
                </div>

                {/* Right Column: Project Metadata */}
                <div className="space-y-3.5">
                  <div>
                    <label className="text-xs font-bold text-slate-300 block mb-1">
                      Project Name
                    </label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. Neon Horizon EP"
                      value={newProject.name}
                      onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                      className="w-full bg-[#222533] border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-bold text-slate-300 block mb-1">
                      Description
                    </label>
                    <div className="relative">
                      <textarea
                        rows={2}
                        maxLength={250}
                        placeholder="Add a description for your project..."
                        value={newProject.description}
                        onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                        className="w-full bg-[#222533] border border-white/10 rounded-xl p-2.5 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-teal-500 transition-colors resize-none pb-5"
                      />
                      <span className="absolute bottom-1.5 right-2.5 text-[9px] font-mono text-slate-500 pointer-events-none">
                        {(newProject.description || '').length}/250
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs font-bold text-slate-300 block mb-1">
                        Tempo (BPM)
                      </label>
                      <input
                        type="number"
                        min={40}
                        max={240}
                        value={newProject.bpm || 120}
                        onChange={(e) => setNewProject({ ...newProject, bpm: parseInt(e.target.value) || 120 })}
                        className="w-full bg-[#222533] border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-bold text-slate-300 block mb-1">
                        Musical Key
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. C Major, A Minor"
                        value={newProject.key_signature || 'C Major'}
                        onChange={(e) => setNewProject({ ...newProject, key_signature: e.target.value })}
                        className="w-full bg-[#222533] border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-xs font-bold text-slate-300 block mb-1">
                      Default Style / Tags
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. Pop, Synthwave, Electronic"
                      value={newProject.tags}
                      onChange={(e) => setNewProject({ ...newProject, tags: e.target.value })}
                      className="w-full bg-[#222533] border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-bold text-slate-300 block mb-1.5">
                      Accent Color
                    </label>
                    <div className="flex gap-2.5">
                      {['teal', 'cyan', 'amber', 'emerald', 'sky'].map((c) => (
                        <button
                          key={c}
                          type="button"
                          onClick={() => setNewProject({ ...newProject, color: c })}
                          className={`w-6 h-6 rounded-full border-2 transition-transform ${
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
                              ? 'border-white scale-110 shadow-md ring-2 ring-white/20'
                              : 'border-transparent opacity-60 hover:opacity-100'
                          }`}
                        />
                      ))}
                    </div>
                  </div>

                  <div className="flex justify-end pt-2">
                    <button
                      type="submit"
                      disabled={!newProject.name.trim()}
                      className="px-6 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-slate-950 font-bold text-xs shadow-md transition-all active:scale-95"
                    >
                      Create Project
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Global Creator Footer */}
      <AppFooter />
    </div>
  );
};
