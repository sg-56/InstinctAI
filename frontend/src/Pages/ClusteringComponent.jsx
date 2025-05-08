"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { useLocation, useParams, useNavigate } from "react-router-dom";
import { ArrowBigDownDash, ChevronRight } from "lucide-react";
import ClusterHistorySection from "../Components/ClusterHistorySection";
import ClusterTreeVisualization from "../Components/ClusterTreeVisualization";
import { useDispatch, useSelector } from "react-redux";
import {
  setClusterHistory,
  setClusters,
  setSelectedIndex,
} from "../redux/clusterSlice";
import { CircularProgress, Typography, Box } from "@mui/material";
import ClusterDropdown from "../Components/ClusterDropdown";
import WorkbenchModal from "../Components/WorkbenchModal";
import DefinationModel from "../Components/DefinationModel";
import SelectableClusterPopup from "../Components/SelectableClustorPopup";

const ClusteringComponent = () => {
  const location = useLocation();
  const baseUrl = import.meta.env.VITE_BASE_URL;
  const { project_id, com_id } = useParams();
  const { activeKPI, kpiList, importantColumnNames } = location.state || {};
  const [newkpi, setNewkpi] = useState(
    activeKPI || (kpiList && kpiList.length > 0 ? kpiList[0] : "")
  );
  const [extractedClusters, setExtractedClusters] = useState([]);
  const [currentPath, setCurrentPath] = useState([]);
  const [error, setError] = useState(null);
  const [currentLevel, setCurrentLevel] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedCell, setSelectedCell] = useState(null);
  const [openDropdowns, setOpenDropdowns] = useState({});
  const [showModal, setShowModal] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [isOpen1, setIsOpen1] = useState(false);
  const [selectedCluster, setSelectedCluster] = useState(0);
  const [groupedClusters, setGroupedClusters] = useState({
    top1: {},
    mean: {},
    percentage: {},
    top2: {},
    top3: {},
    lowest: {},
  });
  const [clusterTree, setClusterTree] = useState(null);
  const [viewMode, setViewMode] = useState("table"); // Default to table view
  const dispatch = useDispatch();
  const { clusterHistory, selectedIndex } = useSelector(
    (state) => state.cluster
  );
  const navigate = useNavigate();
  const [breadcrumbPath, setBreadcrumbPath] = useState([]);
  const [selectedClusterIndex, setSelectedClusterIndex] = useState(null);

  // Add useEffect to call process API on mount and when KPI changes
  useEffect(() => {
    if (newkpi && project_id && !clusterTree) {
      processData(newkpi).catch((error) => {
        console.error("Failed to process data:", error);
        setError(error.message || "Failed to process data");
        setLoading(false);
      });
    }
    // eslint-disable-next-line
  }, [newkpi, project_id]);

  // Data transformation functions
  const transformClusterData = (clusters) => {
    return clusters.reduce(
      (acc, cluster, clusterIndex) => {
        Object.entries(cluster.analysis || {}).forEach(([feature, data]) => {
          // Handle numerical features
          if (data.segment && typeof data.segment.mean === "number") {
            if (!acc.mean[feature]) acc.mean[feature] = {};
            acc.mean[feature][clusterIndex] = {
              original: {
                Mean: data.segment.mean,
                Count: cluster.size || 0,
              },
            };
          }

          // Handle categorical features
          if (data.mode) {
            if (!acc.top1[feature]) acc.top1[feature] = {};
            acc.top1[feature][clusterIndex] = {
              original: {
                Value: data.mode.category,
                Percentage: data.mode.percentage,
              },
            };

            // Handle top categories
            if (data.top_categories && data.top_categories.length > 0) {
              data.top_categories.forEach((cat, idx) => {
                const statKey =
                  idx === 0 ? "top1" : idx === 1 ? "top2" : "top3";
                if (!acc[statKey][feature]) acc[statKey][feature] = {};
                acc[statKey][feature][clusterIndex] = {
                  original: {
                    Value: cat.category,
                    Percentage: cat.percentage,
                  },
                };
              });
            }
          }
        });
        return acc;
      },
      { top1: {}, mean: {}, percentage: {}, top2: {}, top3: {}, lowest: {} }
    );
  };

  const processData = async (kpi) => {
    try {
      setLoading(true);
      console.log("Making process request to backend");
      const response = await axios.post(`${baseUrl}/process/${project_id}`, {
        project_id,
        kpi,
        important_column_names: importantColumnNames,
      });
      console.log("Process response received:", response.data);

      if (response.status === 200 && response.data) {
        const clusterTree = response.data.cluster_tree;
        console.log("Cluster tree:", clusterTree);
        setClusterTree(clusterTree);

        if (!clusterTree || !clusterTree[kpi]) {
          throw new Error(`No clusters found for KPI: ${kpi}`);
        }

        const rootCluster = clusterTree[kpi];
        const currentLevelClusters = getClustersForLevel(
          rootCluster,
          currentLevel
        );

        // Transform the new format to old format
        const transformedData = transformClusterData(currentLevelClusters);
        setGroupedClusters(transformedData);

        setExtractedClusters(currentLevelClusters);
        dispatch(setClusters({ [kpi]: currentLevelClusters }));
        setLoading(false);
      } else {
        throw new Error(`Invalid response: ${response.status}`);
      }
    } catch (error) {
      console.error("Error processing data:", error);
      setError(error.message || "Failed to process data");
      setLoading(false);
    }
  };

  // Helper function to get clusters for a specific level
  const getClustersForLevel = (cluster, targetLevel, currentLevel = 0) => {
    if (currentLevel === targetLevel) {
      return [cluster];
    }

    if (cluster.children && cluster.children.length > 0) {
      return cluster.children.flatMap((child) =>
        getClustersForLevel(child, targetLevel, currentLevel + 1)
      );
    }

    return [];
  };

  useEffect(() => {
    if (clusterHistory.length === 0 && selectedIndex === -1) {
      dispatch(setClusterHistory([]));
    }
  }, []);

  useEffect(() => {
    if (clusterHistory.length > 0) {
      setCurrentLevel(clusterHistory[clusterHistory?.length - 1].level + 1);
    }
  }, [clusterHistory]);

  const handleKpiClick = (kpi) => {
    setBreadcrumbPath([]);
    setSelectedClusterIndex(null);
    setNewkpi(kpi);
    // Update groupedClusters for the new KPI
    if (clusterTree && clusterTree[kpi] && clusterTree[kpi].children) {
      setGroupedClusters(transformClusterData(clusterTree[kpi].children));
    }
  };

  const handleCellClick = (feature, clusterIndex, value) => {
    setSelectedCell({ feature, clusterIndex, currentLevel, value });
    setOpenDropdowns({});
  };

  const toggleDropdown = (e, feature, clusterIndex) => {
    e.stopPropagation();
    setOpenDropdowns((prev) => ({
      ...prev,
      [`${feature}-${clusterIndex}`]: !prev[`${feature}-${clusterIndex}`],
    }));
  };

  const navigateToPath = (path) => {
    setBreadcrumbPath(path);
    setSelectedClusterIndex(null);
    // Update groupedClusters for the new level
    const newNode =
      path.length === 0
        ? clusterTree[newkpi]
        : getClusterByPath(clusterTree[newkpi], path);
    if (newNode && newNode.children) {
      setGroupedClusters(transformClusterData(newNode.children));
    }
    // Update clusterHistory to match the path
    const newHistory = path.map((idx, i) => ({
      path: path.slice(0, i + 1),
      kpi: newkpi,
      level: i + 1,
      cluster: `Cluster ${idx + 1}`,
    }));
    dispatch(setClusterHistory(newHistory));
    dispatch(setSelectedIndex(path.length - 1));
  };

  const handleAnalyze = () => {
    if (selectedClusterIndex !== null) {
      const newPath = [...breadcrumbPath, selectedClusterIndex];
      navigateToPath(newPath);
    }
  };

  const handleHistoryClick = (historyItem, index) => {
    setCurrentLevel(historyItem.level);
    setCurrentPath(historyItem.path);
    dispatch(setSelectedIndex(index));
    setExtractedClusters(historyItem.extractedClusters);
    processData(newkpi);
  };

  const handleDownload = async (clusterIndex) => {
    try {
      const response = await axios.post(
        `${baseUrl}/projects/${project_id}/clusters/download`,
        {
          project_id,
          level: currentLevel,
          path: currentPath,
          cluster_index: clusterIndex,
        },
        {
          responseType: "blob",
        }
      );
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `cluster_${clusterIndex + 1}_level_${currentLevel}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Error downloading cluster:", err);
      setError("Failed to download cluster data");
    }
  };

  const filterImportantFeatures = (features) => {
    return features.filter((feature) => importantColumnNames.includes(feature));
  };

  const getClusterByPath = (tree, path) => {
    let node = tree;
    for (const idx of path) {
      if (!node.children || !node.children[idx]) return null;
      node = node.children[idx];
    }
    return node;
  };

  const handleColumnHeaderClick = (clusterIndex) => {
    setSelectedClusterIndex(clusterIndex);
  };

  const handleBreadcrumbClick = (level) => {
    const newPath = breadcrumbPath.slice(0, level + 1);
    setBreadcrumbPath(newPath);
    setSelectedClusterIndex(null);

    // Update groupedClusters for the new level
    const newNode =
      newPath.length === 0
        ? clusterTree[newkpi]
        : getClusterByPath(clusterTree[newkpi], newPath);
    if (newNode && newNode.children) {
      setGroupedClusters(transformClusterData(newNode.children));
    }
    // Update clusterHistory for Segment History
    const newHistory = clusterHistory.slice(0, level + 1);
    dispatch(setClusterHistory(newHistory));
    dispatch(setSelectedIndex(level));
  };

  const handleDownloadCSV = async () => {
    try {
      // Convert indexes to strings for backend compatibility
      const indexes = (breadcrumbPath.length > 0 ? breadcrumbPath : [0]).map(
        String
      );
      const response = await axios.post(
        `${baseUrl}/projects/${project_id}/clusters/get_clusters`,
        indexes,
        { headers: { "Content-Type": "application/json" } }
      );
      const html = response.data;
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, "text/html");
      const table = doc.querySelector("table");
      let csv = "";
      for (const row of table.rows) {
        const cells = Array.from(row.cells).map(
          (cell) => '"' + cell.innerText.replace(/"/g, '""') + '"'
        );
        csv += cells.join(",") + "\n";
      }
      const blob = new Blob([csv], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `clusters_level_${breadcrumbPath.length}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError("Failed to download CSV");
    }
  };

  // Add useEffect to update groupedClusters when clusterTree or newkpi changes (initial load or KPI change)
  useEffect(() => {
    if (
      clusterTree &&
      newkpi &&
      clusterTree[newkpi] &&
      clusterTree[newkpi].children
    ) {
      setGroupedClusters(transformClusterData(clusterTree[newkpi].children));
    }
  }, [clusterTree, newkpi]);

  if (!project_id || !location.state) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography color="error">
          Project not found or no change made
        </Typography>
      </Box>
    );
  }

  // Show loader while loading
  if (loading) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        gap={2}
      >
        <CircularProgress size={60} />
        <Typography variant="h6" color="textSecondary">
          Loading cluster data...
        </Typography>
      </Box>
    );
  }

  // Show error if there is one
  if (error) {
    return (
      <div className="flex h-[calc(100vh-80px)] items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-red-600 mb-4">Error</h2>
          <p className="text-gray-700">{error}</p>
          <div className="mt-4 space-x-4">
            <button
              onClick={() => {
                setError(null);
                setLoading(false);
              }}
              className="px-4 py-2 bg-purple-400 text-white rounded-md hover:bg-purple-500"
            >
              Try Again
            </button>
            <button
              onClick={() => {
                window.location.href = `/${com_id}/projects/${project_id}/configuration/`;
              }}
              className="px-4 py-2 bg-gray-400 text-white rounded-md hover:bg-gray-500"
            >
              Go to Configuration
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Show "No data" message only if we're not loading and there's no error
  if (!extractedClusters || extractedClusters.length === 0) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography>No cluster data available</Typography>
      </Box>
    );
  }

  const currentNode =
    breadcrumbPath.length === 0
      ? clusterTree[newkpi]
      : getClusterByPath(clusterTree[newkpi], breadcrumbPath);
  const currentClusters =
    currentNode && currentNode.children ? currentNode.children : [];

  return (
    <div className="flex h-[calc(100vh-80px)] overflow-hidden overflow-y-auto">
      <div className="w-64 bg-gray-100 border-r border-gray-200 overflow-y-auto">
        <ClusterHistorySection
          selectedCell={selectedCell}
          currentLevel={breadcrumbPath.length}
          selectedIndex={breadcrumbPath.length - 1}
          clusterHistory={clusterHistory}
          onSegmentClick={navigateToPath}
        />
      </div>
      <div className="w-[calc(100%-16rem)] overflow-hidden overflow-y-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-xl font-semibold text-gray-700">
            Select the target KPI for analysis
          </h1>
          <div className="flex items-center space-x-2">
            <button
              onClick={() =>
                setViewMode(viewMode === "table" ? "tree" : "table")
              }
              className="px-4 py-2 text-sm font-medium rounded-md bg-white border border-gray-200 hover:bg-gray-50"
            >
              Switch to {viewMode === "table" ? "Tree" : "Table"} View
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-8">
          {kpiList.map((kpi, index) => (
            <button
              key={index}
              onClick={() => handleKpiClick(kpi)}
              className={`px-4 py-2 text-sm font-medium rounded-md min-w-[100px] ${
                newkpi === kpi
                  ? "text-white bg-purple-400"
                  : "text-gray-600 bg-white hover:bg-gray-50"
              }`}
            >
              {kpi}
            </button>
          ))}
        </div>

        {viewMode === "table" ? (
          <div className="flex flex-row justify-start items-start gap-0 w-min mx-auto">
            <div className="flex flex-col justify-start mx-auto items-start p-1 w-fit">
              <div className="flex items-center gap-2 mb-4 text-sm font-medium">
                <span className="text-gray-500">KPI:</span>
                <span className="text-indigo-700 font-bold mr-2">{newkpi}</span>
                <button
                  className={`text-blue-600 hover:underline ${
                    breadcrumbPath.length === 0 ? "font-bold" : ""
                  }`}
                  onClick={() => {
                    navigateToPath([]);
                  }}
                >
                  Root
                </button>
                {breadcrumbPath.map((idx, i) => (
                  <span key={`bc-${i}`} className="flex items-center">
                    <ChevronRight className="inline-block w-4 h-4 mx-1 text-gray-400" />
                    <button
                      className={`text-blue-600 hover:underline ${
                        i === breadcrumbPath.length - 1 ? "font-bold" : ""
                      }`}
                      onClick={() =>
                        navigateToPath(breadcrumbPath.slice(0, i + 1))
                      }
                    >
                      Cluster {idx + 1}
                    </button>
                  </span>
                ))}
              </div>
              <div className="w-full overflow-x-auto">
                <div className="w-[calc(100vw-18rem)] overflow-x-auto border-b border-gray-200 shadow sm:rounded-lg">
                  {loading && (
                    <Box
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      p={4}
                    >
                      <CircularProgress />
                    </Box>
                  )}
                  {error && <p className="text-red-500 p-4">Error: {error}</p>}
                  {!loading && !error && currentClusters.length > 0 && (
                    <table className="w-[calc(100vw-20rem)] divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th
                            scope="col"
                            className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky left-0 bg-gray-50"
                          >
                            Segments/Parameters
                          </th>
                          {currentClusters.map((_, index) => (
                            <th
                              key={index}
                              scope="col"
                              className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer ${
                                selectedClusterIndex === index
                                  ? "bg-indigo-200"
                                  : ""
                              }`}
                              onClick={() => handleColumnHeaderClick(index)}
                            >
                              <div className="flex items-center justify-start min-w-[120px] max-w-[120px]">
                                Segment {index + 1}
                                <button
                                  className="ml-2 p-1 rounded-full hover:bg-gray-200 transition-colors"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    handleDownload(index);
                                  }}
                                >
                                  <ArrowBigDownDash className="w-4 h-4" />
                                </button>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {filterImportantFeatures(
                          Object.keys(groupedClusters.top1)
                        ).map((feature) => (
                          <tr key={feature} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 sticky left-0 bg-white">
                              {feature}
                            </td>
                            {currentClusters.map((_, clusterIndex) => (
                              <td
                                key={clusterIndex}
                                className={`px-6 py-4 whitespace-nowrap text-sm ${
                                  selectedCell?.feature === feature &&
                                  selectedCell?.clusterIndex === clusterIndex
                                    ? "bg-indigo-100"
                                    : ""
                                }`}
                              >
                                <div
                                  onClick={() =>
                                    handleCellClick(
                                      feature,
                                      clusterIndex,
                                      groupedClusters.top1?.[feature]?.[
                                        clusterIndex
                                      ]?.original?.Value ??
                                        groupedClusters.mean?.[feature]?.[
                                          clusterIndex
                                        ]?.original?.Mean ??
                                        0
                                    )
                                  }
                                  className="cursor-pointer hover:bg-indigo-50 p-2 rounded transition-colors hover:underline"
                                >
                                  {groupedClusters.top1?.[feature]?.[
                                    clusterIndex
                                  ]?.original?.Value ??
                                    groupedClusters.mean?.[feature]?.[
                                      clusterIndex
                                    ]?.original?.Mean ??
                                    0}
                                  <span className="ml-2 text-sm text-gray-500">
                                    -{" "}
                                    {groupedClusters.top1[feature][clusterIndex]
                                      ?.original.Percentage || 0}{" "}
                                    <span
                                      className="pl-4"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        toggleDropdown(
                                          e,
                                          feature,
                                          clusterIndex
                                        );
                                      }}
                                    >
                                      ▼
                                    </span>
                                  </span>
                                </div>
                                {openDropdowns[
                                  `${feature}-${clusterIndex}`
                                ] && (
                                  <ClusterDropdown
                                    groupedClusters={groupedClusters}
                                    feature={feature}
                                    handleCellClick={handleCellClick}
                                    toggleDropdown={toggleDropdown}
                                    clusterIndex={clusterIndex}
                                  />
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                        {filterImportantFeatures(
                          Object.keys(groupedClusters.mean)
                        ).map((feature) => (
                          <tr key={feature} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 sticky left-0 bg-white">
                              {feature}
                            </td>
                            {currentClusters.map((_, clusterIndex) => (
                              <td
                                key={clusterIndex}
                                className={`px-6 py-4 whitespace-nowrap text-sm ${
                                  selectedCell?.feature === feature &&
                                  selectedCell?.clusterIndex === clusterIndex
                                    ? "bg-indigo-100"
                                    : ""
                                }`}
                                onClick={() =>
                                  handleCellClick(
                                    feature,
                                    clusterIndex,
                                    groupedClusters.top1?.[feature]?.[
                                      clusterIndex
                                    ]?.original?.Value ??
                                      groupedClusters.mean?.[feature]?.[
                                        clusterIndex
                                      ]?.original?.Mean ??
                                      0
                                  )
                                }
                                onDoubleClick={() => {
                                  setIsOpen1(true);
                                  setSelectedCluster(clusterIndex + 1);
                                }}
                              >
                                <div
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleDropdown(e, feature, clusterIndex);
                                  }}
                                  className="cursor-pointer hover:bg-indigo-50 p-2 rounded transition-colors"
                                >
                                  {groupedClusters.top1?.[feature]?.[
                                    clusterIndex
                                  ]?.original?.Mean?.toFixed(4) ??
                                    groupedClusters.mean?.[feature]?.[
                                      clusterIndex
                                    ]?.original?.Mean?.toFixed(4) ??
                                    0}
                                  <span className="ml-2 text-sm text-gray-500">
                                    (
                                    {groupedClusters.mean[feature][clusterIndex]
                                      ?.original.Count || 0}
                                    )
                                  </span>
                                </div>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap justify-start items-center gap-4 mt-4">
                <button
                  onClick={handleAnalyze}
                  disabled={selectedClusterIndex === null}
                  className={`px-4 py-2 rounded ${
                    selectedClusterIndex !== null
                      ? "bg-blue-600 text-white hover:bg-blue-700"
                      : "bg-gray-300 text-gray-500 cursor-not-allowed"
                  }`}
                >
                  Analyze
                </button>
                <button
                  onClick={() =>
                    navigate(`/projects/${project_id}/workbench`, {
                      state: {
                        activeKPI: newkpi,
                        kpiList,
                        importantColumnNames,
                      },
                    })
                  }
                  className="p-2 rounded-lg font-semibold border px-4 bg-white text-gray-800 hover:bg-gray-100 transition-colors"
                >
                  Workbench
                </button>
              </div>
            </div>
          </div>
        ) : (
          <ClusterTreeVisualization
            clusterTree={clusterTree}
            activeKPI={newkpi}
            onClusterSelect={(cluster, path, level) => {
              setCurrentLevel(level);
              setCurrentPath(path);
              if (cluster.children && cluster.children.length > 0) {
                setExtractedClusters(cluster.children);
                const transformedData = transformClusterData(cluster.children);
                setGroupedClusters(transformedData);
              }
            }}
          />
        )}

        {showModal && (
          <WorkbenchModal
            categorical_columns={[]}
            currentLevel={currentLevel}
            currentPath={currentPath}
            activeKPI={newkpi}
            showModal={showModal}
            setShowModal={setShowModal}
            project_id={project_id}
          />
        )}
        {isOpen && (
          <DefinationModel
            setIsOpen={setIsOpen}
            kpi={newkpi}
            clusterNo={selectedCluster}
            path={currentPath}
          />
        )}
        {isOpen1 && (
          <SelectableClusterPopup
            setIsOpen1={setIsOpen1}
            selectedCluster={selectedCluster}
            setSelectedCluster={setSelectedCluster}
            kpi={newkpi}
          />
        )}

        <button
          className="mb-4 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          onClick={handleDownloadCSV}
        >
          Download CSV
        </button>
      </div>
    </div>
  );
};

export default ClusteringComponent;
