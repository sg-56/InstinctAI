"use client";

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  Folder,
  FolderOpen,
  AlertCircle,
  RefreshCw,
  Search,
  Edit,
  Trash2,
  X,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import CreateProjModal from "../Components/CreateProjModal";
import Loader from "../Components/Loader";
import { createProject } from "../redux/projectsSlice";
import axiosInstance from "../utils/axiosInstance";
import { getAuthState } from "../utils/auth";

const Home = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [newProject, setNewProject] = useState({ name: "", description: "" });
  const baseUrl = import.meta.env.VITE_BASE_URL;
  const { com_id } = getAuthState();
  const projectsInStore = useSelector((state) => state.projects);
  const [searchTerm, setSearchTerm] = useState("");
  const hasFetched = useRef(false);

  // Add state for project actions
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [projectToModify, setProjectToModify] = useState(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const toastTimer = useRef(null);

  useEffect(() => {
    if (!com_id) {
      setError("Company ID is not set. Please log in again.");
      return;
    }

    // Only load projects if we haven't fetched them yet
    if (!hasFetched.current) {
      loadProjects();
      hasFetched.current = true;
    }
  }, [com_id]);

  useEffect(() => {
    if (showSuccess) {
      toastTimer.current = setTimeout(() => {
        setShowSuccess(false);
      }, 5000);
    }
    return () => {
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, [showSuccess]);

  const loadProjects = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axiosInstance.get(`/${com_id}/projects/`);
      const fetchedProjects = response.data.projects || [];

      // Transform the projects to match the expected structure
      const transformedProjects = fetchedProjects.map((project) => ({
        project_id: project.project_id,
        name: project.name,
        description: project.description,
        com_id: com_id,
      }));

      // Update Redux store only for new projects
      transformedProjects.forEach((project) => {
        const projectExists = projectsInStore.some(
          (p) => p.projectId === project.project_id
        );

        if (!projectExists) {
          dispatch(
            createProject({
              projectId: project.project_id,
              name: project.name,
              description: project.description,
              columns: [],
              importantColumnNames: [],
              kpiList: [],
              droppedColumns: [],
              uploadedFileData: [],
              selectedKpi: null,
            })
          );
        }
      });

      setProjects(transformedProjects);
    } catch (error) {
      console.error("Error fetching projects:", error);
      setError("Failed to fetch projects. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleNewProject = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);

      if (!newProject.name.trim()) {
        throw new Error("Project name is required");
      }

      const response = await axiosInstance.post(`/${com_id}/projects/`, {
        name: newProject.name,
        description: newProject.description,
      });
      const projectId = response?.data?.project_id;

      if (!projectId) {
        throw new Error("Invalid project ID received from server");
      }

      localStorage.setItem("project_id", projectId);
      dispatch(
        createProject({
          projectId,
          columns: [],
          importantColumnNames: [],
          kpiList: [],
          droppedColumns: [],
          uploadedFileData: [],
          selectedKpi: null,
        })
      );

      // Show success message
      setSuccessMessage("Project created successfully!");
      setShowSuccess(true);
      setShowModal(false);
      setNewProject({ name: "", description: "" });

      // Wait for animation to complete before navigating
      setTimeout(() => {
        navigate(`/${com_id}/projects/${projectId}/configuration`);
      }, 1500);
    } catch (error) {
      console.error("Error creating project:", error);
      setError(
        error.response?.data?.message ||
          error.message ||
          "Failed to create project"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleProjectStatus = async (projectId) => {
    try {
      if (!projectId) {
        throw new Error("Invalid project ID");
      }

      const response = await axios.get(
        `${baseUrl}/projects/${projectId}/status`
      );

      if (!response.data) {
        throw new Error("Invalid response from server");
      }

      return response;
    } catch (error) {
      if (error.response) {
        if (error.response.status === 404) {
          throw new Error("Project not found or data not available");
        } else {
          throw new Error(
            `Server error: ${error.response.data?.message || "Unknown error"}`
          );
        }
      }
      throw new Error(
        "Failed to fetch project status: " + (error.message || "Unknown error")
      );
    }
  };

  const handleProjectClick = async (projectId) => {
    try {
      if (!projectId) {
        throw new Error("Invalid project ID");
      }

      setLoading(true);
      setError(null);
      localStorage.setItem("project_id", projectId);

      let status;
      try {
        status = await handleProjectStatus(projectId);
      } catch (statusError) {
        console.warn(
          "Status check failed, redirecting to upload:",
          statusError
        );
        navigate(`/${com_id}/projects/${projectId}/configuration`);
        setLoading(false);
        return;
      }

      if (!status.data.data_uploaded) {
        navigate(`/${com_id}/projects/${projectId}/configuration`);
        setLoading(false);
        return;
      }

      if (status.data.feature_ranking_completed) {
        navigate(`/${com_id}/projects/${projectId}/clustered-data`, {
          state: {
            kpiList: status.data.kpi_list,
            importantColumnNames: status.data.important_features,
            activeKPI: status.data.kpi,
            rankingStatus: "SUCCESS",
          },
        });
      } else {
        try {
          const rankingResponse = await axios.post(
            `${baseUrl}/projects/${projectId}/feature-ranking`,
            {
              project_id: projectId,
              kpi: status.data.kpi,
              important_features: status.data.important_features,
              kpi_list: status.data.kpi_list,
            }
          );

          navigate(`/${com_id}/projects/${projectId}/clustered-data`, {
            state: {
              kpiList: status.data.kpi_list,
              importantColumnNames: status.data.important_features,
              activeKPI: status.data.kpi,
              rankingStatus: false,
              task_id: rankingResponse.data.task_id,
            },
          });
        } catch (rankingError) {
          console.error("Feature ranking failed:", rankingError);
          throw new Error("Failed to start feature ranking process");
        }
      }
    } catch (error) {
      console.error("Error navigating to project:", error);
      setError(error.message || "Failed to open project. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Filter projects based on search term
  const filteredProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 300, damping: 24 },
    },
  };

  const handleRenameProject = async () => {
    if (!projectToModify || !newProjectName.trim()) {
      console.warn("No project selected for rename.");
      return;
    }
    console.log("Renaming project with ID:", projectToModify.project_id);
    try {
      setLoading(true);
      setError(null);
      await axiosInstance.put(
        `/${com_id}/projects/${projectToModify.project_id}`,
        {
          name: newProjectName.trim(),
        }
      );
      setShowRenameModal(false);
      setSuccessMessage("Project renamed successfully!");
      setShowSuccess(true);
      setProjectToModify(null);
      setNewProjectName("");
      await loadProjects();
    } catch (error) {
      setError(error.response?.data?.message || "Failed to rename project.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!projectToModify) {
      console.warn("No project selected for delete.");
      return;
    }
    console.log("Deleting project with ID:", projectToModify.project_id);
    try {
      setLoading(true);
      setError(null);
      await axiosInstance.delete(
        `/${com_id}/projects/${projectToModify.project_id}`
      );
      setShowDeleteConfirm(false);
      setSuccessMessage("Project deleted successfully!");
      setShowSuccess(true);
      setProjectToModify(null);
      await loadProjects();
    } catch (error) {
      setError(error.response?.data?.message || "Failed to delete project.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Projects
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage and analyze your data projects
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
          <div className="relative w-full sm:w-64">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search projects..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input pl-10 w-full"
            />
          </div>

          <motion.button
            onClick={() => setShowModal(true)}
            className="btn btn-primary flex items-center justify-center gap-2 px-4 py-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="h-5 w-5" />
            <span>New Project</span>
          </motion.button>
        </div>
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 rounded-lg bg-red-50 border-l-4 border-red-500 dark:bg-red-900/20 dark:border-red-500"
        >
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800 dark:text-red-400">
                {error}
              </p>
            </div>
            <motion.button
              onClick={loadProjects}
              disabled={loading}
              className="btn btn-sm btn-outline flex items-center gap-2 text-red-600 border-red-300 hover:bg-red-50 dark:text-red-400 dark:border-red-700 dark:hover:bg-red-900/20"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <RefreshCw className="h-4 w-4" />
              <span>Retry</span>
            </motion.button>
          </div>
        </motion.div>
      )}

      {loading ? (
        <Loader text="Loading projects..." />
      ) : filteredProjects.length > 0 ? (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
        >
          {filteredProjects.map((project) => (
            <motion.div
              key={project.project_id}
              variants={itemVariants}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
              className="group"
            >
              <div className="h-full bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden cursor-pointer border border-gray-200 dark:border-gray-700">
                <div className="h-32 bg-gradient-to-r from-indigo-500 to-purple-600 relative overflow-hidden">
                  <motion.div
                    className="absolute inset-0 flex items-center justify-center"
                    initial={false}
                    whileHover={{ scale: 1.2, rotate: 15 }}
                  >
                    <FolderOpen className="h-20 w-20 text-white/20 group-hover:text-white/30 transition-colors" />
                  </motion.div>
                  <div className="absolute top-2 right-2 flex gap-1">
                    <motion.button
                      onClick={(e) => {
                        e.stopPropagation();
                        setProjectToModify(project);
                        setNewProjectName(project.name);
                        setShowRenameModal(true);
                      }}
                      className="p-1.5 rounded-full bg-white/20 hover:bg-white/40 text-white transition-colors"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <Edit className="h-3.5 w-3.5" />
                    </motion.button>
                    <motion.button
                      onClick={(e) => {
                        e.stopPropagation();
                        setProjectToModify(project);
                        setShowDeleteConfirm(true);
                      }}
                      className="p-1.5 rounded-full bg-white/20 hover:bg-red-400 text-white transition-colors"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </motion.button>
                  </div>
                </div>
                <div
                  className="p-5"
                  onClick={() =>
                    !loading && handleProjectClick(project.project_id)
                  }
                >
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white capitalize mb-2 truncate">
                    {project.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 h-10">
                    {project.description || "No description provided"}
                  </p>
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      Created:{" "}
                      {new Date(
                        project.createdAt || Date.now()
                      ).toLocaleDateString()}
                    </span>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                      Active
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-center py-16 bg-gray-50 dark:bg-gray-800/50 rounded-xl"
        >
          <Folder className="h-16 w-16 mx-auto text-gray-400 dark:text-gray-600 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No projects found
          </h3>
          <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto mb-6">
            {searchTerm
              ? "No projects match your search criteria."
              : "You haven't created any projects yet. Create your first project to get started."}
          </p>
          <motion.button
            onClick={() => setShowModal(true)}
            className="btn btn-primary inline-flex items-center gap-2 px-4 py-3"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="h-5 w-5" />
            <span>Create Your First Project</span>
          </motion.button>
        </motion.div>
      )}

      {showModal && (
        <CreateProjModal
          setNewProject={setNewProject}
          setShowModal={setShowModal}
          handleNewProject={handleNewProject}
          newProject={newProject}
          loading={loading}
        />
      )}

      {showRenameModal && (
        <motion.div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setShowRenameModal(false)}
        >
          <motion.div
            className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden"
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{
              opacity: 1,
              y: 0,
              scale: 1,
              transition: { type: "spring", damping: 25, stiffness: 300 },
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Rename Project
              </h2>
              <motion.button
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowRenameModal(false)}
              >
                <X className="h-5 w-5" />
              </motion.button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleRenameProject();
              }}
              className="p-6"
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Project Name
                  </label>
                  <input
                    type="text"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    className="input w-full"
                    placeholder="Enter new project name"
                    required
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <motion.button
                  type="button"
                  onClick={() => setShowRenameModal(false)}
                  className="btn btn-sm btn-outline"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Cancel
                </motion.button>
                <motion.button
                  type="submit"
                  className="btn btn-sm btn-primary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Rename Project
                </motion.button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}

      {showDeleteConfirm && (
        <motion.div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setShowDeleteConfirm(false)}
        >
          <motion.div
            className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden"
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{
              opacity: 1,
              y: 0,
              scale: 1,
              transition: { type: "spring", damping: 25, stiffness: 300 },
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Delete Project
              </h2>
              <motion.button
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowDeleteConfirm(false)}
              >
                <X className="h-5 w-5" />
              </motion.button>
            </div>

            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-full">
                  <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    Confirm deletion
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Are you sure you want to delete{" "}
                    <span className="font-medium">{projectToModify?.name}</span>
                    ? This action cannot be undone.
                  </p>
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <motion.button
                  type="button"
                  onClick={() => setShowDeleteConfirm(false)}
                  className="btn btn-sm btn-outline"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Cancel
                </motion.button>
                <motion.button
                  type="button"
                  onClick={handleDeleteProject}
                  className="btn btn-sm bg-red-600 hover:bg-red-700 text-white"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Delete Project
                </motion.button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.3 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5, transition: { duration: 0.2 } }}
            className="fixed top-4 right-4 z-50"
          >
            <div className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-2">
              <CheckCircle className="h-5 w-5" />
              <span>{successMessage}</span>
            </div>
            <button
              className="ml-2 text-white/80 hover:text-white"
              onClick={() => {
                setShowSuccess(false);
                if (toastTimer.current) clearTimeout(toastTimer.current);
              }}
            >
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Home;
