"use client";

import { useState, useEffect } from "react";
import { useLocation, useParams } from "react-router-dom";
import axios from "axios";
import ClusterHistorySection from "../Components/ClusterHistorySection";
import { featureRanking } from "../utils/apiUtils";
import { ArrowDown, ChevronDownIcon, Command } from "lucide-react";
import ClusterDropdown from "../Components/ClusterDropdown";
import WorkbenchModal from "../Components/WorkbenchModal";

export default function AnalysisDashboard() {
  const location = useLocation();
  const baseUrl = import.meta.env.VITE_BASE_URL;
  const { project_id } = useParams();
  // const { importantColumnNames, kpiList } = location.state || {};
  // const [activeKPI, setActiveKPI] = useState(kpiList[0]);
  const { importantColumnNames = [], kpiList = [] } = location.state || {};
  const [activeKPI, setActiveKPI] = useState(kpiList?.[0] || "");
  const [loading, setLoading] = useState(false);
  const [task_id, setTask_id] = useState("");

  console.log(kpiList);
  console.log(importantColumnNames);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const task_id = await featureRanking(
          project_id,
          activeKPI,
          importantColumnNames,
          kpiList
        );
        setTask_id(task_id);
      } catch (error) {
        console.error("Error:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const [newkpi, setNewkpi] = useState(activeKPI);
  const [allClusters, setAllClusters] = useState([]);
  const [extractedClusters, setExtractedClusters] = useState([]);
  const [currentPath, setCurrentPath] = useState([]);
  const [error, setError] = useState(null);
  const [currentLevel, setCurrentLevel] = useState(0);
  const [openDropdowns, setOpenDropdowns] = useState({});
  const [selectedCell, setSelectedCell] = useState(null);
  const [clusterHistory, setClusterHistory] = useState([]);
  const [categorical_columns1, setCategorical_columns1] = useState([]);
  const MAX_LEVEL = 8;
  const [showModal, setShowModal] = useState(false);

  const toggleDropdown = (e, feature, clusterIndex) => {
    e.stopPropagation();
    setOpenDropdowns((prev) => ({
      ...prev,
      [`${feature}-${clusterIndex}`]: !prev[`${feature}-${clusterIndex}`],
    }));
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

      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob);

      // Create a temporary anchor element
      const link = document.createElement("a");
      link.href = url;
      link.download = `cluster_${clusterIndex + 1}_level_${currentLevel}.csv`;

      // Append to document, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the URL
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Error downloading cluster:", err);
      setError("Failed to download cluster data");
    }
  };

  const fetchClusters = async (level, path) => {
    try {
      setLoading(true);
      setError(null);
      if (!baseUrl || !project_id) {
        throw new Error("Missing base URL or project ID");
      }
      const subclusterResponse = await axios.post(
        `${baseUrl}/projects/${project_id}/clusters/subcluster`,
        {
          project_id,
          kpi: newkpi,
          level,
          path,
        }
      );
      console.log(subclusterResponse);
      const task_id2 = subclusterResponse.data.task_id;

      // Wait for subcluster task to complete
      const checkSubclusterStatus = () => {
        return new Promise((resolve, reject) => {
          const intervalId = setInterval(async () => {
            try {
              const subclusterStatus = await axios.get(
                `${baseUrl}/projects/tasks/${task_id2}/status`
              );
              console.log(subclusterStatus);
              if (subclusterStatus.data.status === "SUCCESS") {
                clearInterval(intervalId);
                resolve();
              } else if (subclusterStatus.data.status === "FAILURE") {
                reject(new Error("Subcluster task failed"));
                clearInterval(intervalId);
                console.log(subclusterResponse);
                throw new Error("Processing subcluster failed");
              }
            } catch (err) {
              clearInterval(intervalId);
              reject(err);
            }
          }, 1000);
        });
      };
      await checkSubclusterStatus();

      // Now fetch the summary
      const summaryResponse = await axios.post(
        `${baseUrl}/projects/${project_id}/clusters/summarize`,
        {
          project_id,
          level,
          path,
        }
      );
      console.log(
        "subclusterResponse",
        subclusterResponse,
        "summaryResponse",
        summaryResponse
      );
      const clusters = summaryResponse.data
        .map((cluster) => {
          const [key] = Object.keys(cluster);
          const index = Number.parseInt(key.split("_")[1], 10);
          return { index, value: cluster[key] };
        })
        .sort((a, b) => a.index - b.index)
        .map((cluster) => cluster.value);

      setExtractedClusters(clusters);

      const processedClusters = clusters.flatMap((clusterArray, clusterIndex) =>
        clusterArray.map((clusterData) => ({
          id: clusterIndex,
          level,
          path,
          original: clusterData,
        }))
      );
      console.log(clusters);
      console.log(processedClusters);
      setAllClusters((prev) => [...prev, ...processedClusters]);
      setLoading(false);
    } catch (err) {
      console.error("Error Details:", err);
      setError(
        err.response?.data?.message || err.message || "Failed to fetch clusters"
      );
      setLoading(false);
    }
  };
  console.log(extractedClusters);

  const filterImportantFeatures = (features) => {
    return features.filter((feature) => importantColumnNames.includes(feature));
  };

  const groupClustersByStatistic = (statistic) => {
    return allClusters
      .filter((cluster) => cluster.original.Statistic === statistic)
      .reduce((acc, cluster) => {
        const feature = cluster.original.Feature;
        if (!acc[feature]) {
          acc[feature] = [];
        }
        acc[feature].push(cluster);
        return acc;
      }, {});
  };

  const groupedClusters = {
    top1: groupClustersByStatistic("Top 1 Value"),
    top2: groupClustersByStatistic("Top 2 Value"),
    top3: groupClustersByStatistic("Top 3 Value"),
    lowest: groupClustersByStatistic("Lowest Value"),
    mean: groupClustersByStatistic("Mean"),
  };

  const handleKpiClick = async (kpi) => {
    setAllClusters([]);
    setCurrentLevel(0);
    setCurrentPath([]);
    setNewkpi(kpi);
    setSelectedCell(null);
    const task_id = await featureRanking(
      project_id,
      kpi,
      importantColumnNames,
      kpiList
    );
    await fetchClusters(currentLevel, currentPath);
  };

  const handleCellClick = (feature, clusterIndex) => {
    setSelectedCell({ feature, clusterIndex, currentLevel });
    setOpenDropdowns({});
  };

  const handleAnalyze = () => {
    if (selectedCell && currentLevel < MAX_LEVEL) {
      const newPath = [...currentPath, selectedCell.clusterIndex];
      setClusterHistory((prev) => [
        ...prev,
        {
          cluster: selectedCell.clusterIndex + 1,
          feature: selectedCell.feature,
          level: currentLevel,
          extractedClusters: extractedClusters,
          path: currentPath,
        },
      ]);

      setCurrentPath(newPath);
      setCurrentLevel(currentLevel + 1);
      setAllClusters([]);
      fetchClusters(currentLevel + 1, newPath);
      setSelectedCell(null);
    }
  };

  const handleHistoryClick = (historyItem) => {
    setCurrentLevel(historyItem.level);
    setCurrentPath(historyItem.path);
    setAllClusters([]);
    setExtractedClusters(historyItem.extractedClusters);
    fetchClusters(historyItem.level, historyItem.path);
  };

  const handleWorkbench = async () => {
    setShowModal(true);
    try {
      const categCols = await axios.post(
        `${baseUrl}/projects/${project_id}/time-series/categorical-columns`,
        {
          project_id,
          level: currentLevel,
          path: currentPath,
        }
      );
      console.log(categCols.data.categorical_columns);
      setCategorical_columns1(categCols.data.categorical_columns);
    } catch (error) {
      console.error("Error fetching categorical columns:", error);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        Loading...
      </div>
    );
  }

  if (error) {
    return <div className="text-red-500 text-center">{error}</div>;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <div className="w-64 bg-gray-100 border-r border-gray-200 overflow-y-auto">
        <ClusterHistorySection
          selectedCell={selectedCell}
          currentLevel={currentLevel}
          clusterHistory={clusterHistory}
          handleHistoryClick={handleHistoryClick}
        />
      </div>
      <div className="flex-1 overflow-auto p-6">
        <div className="flex flex-wrap gap-2 mb-8">
          {kpiList.map((kpi, index) => (
            <button
              key={index}
              onClick={() => handleKpiClick(kpi)}
              className={`px-4 py-2 text-sm font-medium rounded-md min-w-[100px] ${
                activeKPI === kpi
                  ? "text-white bg-purple-400"
                  : "text-gray-600 bg-white hover:bg-gray-50"
              }`}
            >
              {kpi}
            </button>
          ))}
        </div>

        <div className="mb-8 w-full overflow-x-auto bg-white">
          <table className="w-full border-collapse border border-gray-300 min-w-[800px]">
            <thead className="bg-black text-white sticky top-0 z-10">
              <tr>
                <th className="border px-2 py-2 sticky left-0 z-20">
                  Segments/
                  <br />
                  Parameters
                </th>
                {extractedClusters.map((_, index) => (
                  <th
                    key={index}
                    className="border py-2 px-1 text-center whitespace-nowrap"
                  >
                    Segment {index + 1}
                    <button
                      className="p-1 border border-white rounded-full ml-2 hover:bg-indigo-500 transition-colors"
                      onClick={(e) => {
                        e.preventDefault();
                        handleDownload(index);
                      }}
                    >
                      <ArrowDown className="h-4 w-4" />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filterImportantFeatures(Object.keys(groupedClusters.top1)).map(
                (feature) => (
                  <tr key={feature} className="hover:bg-gray-50  bg-white">
                    <td className="border p-2 hover:text-white sticky z-40 bg-white -left-5">
                      {feature}
                    </td>
                    {extractedClusters.map((_, clusterIndex) => (
                      <td
                        key={clusterIndex}
                        className={`border py-1  hover:bg-indigo-400 cursor-pointer ${
                          selectedCell?.feature === feature &&
                          selectedCell?.clusterIndex === clusterIndex
                            ? "bg-indigo-400 "
                            : "bg-white text-black"
                        }`}
                        onClick={() => handleCellClick(feature, clusterIndex)}
                      >
                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleDropdown(e, feature, clusterIndex);
                          }}
                          className="cursor-pointer hover:bg-indigo-400 hover:text-white p-2 flex w-fll justify-between"
                        >
                          {groupedClusters.top1?.[feature]?.[clusterIndex]
                            ?.original?.Value ??
                            groupedClusters.mean?.[feature]?.[clusterIndex]
                              ?.original?.Mean ??
                            0}
                          <br />

                          <span className="ml-2 text-sm text-gray-500 hover:text-white flex">
                            {" "}
                            (
                            {groupedClusters.top1[feature][clusterIndex]
                              ?.original.Count || 0}
                            ) ▼
                          </span>
                        </div>
                        {openDropdowns[`${feature}-${clusterIndex}`] && (
                          <ClusterDropdown
                            groupedClusters={groupedClusters}
                            feature={feature}
                            toggleDropdown={toggleDropdown}
                            clusterIndex={clusterIndex}
                          />
                        )}
                      </td>
                    ))}
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>

        <div className="flex flex-wrap items-center gap-4 mb-8">
          <button
            onClick={handleAnalyze}
            disabled={!selectedCell || currentLevel >= MAX_LEVEL}
            className={`p-2 rounded-lg border px-4 flex flex-row justify-center items-center gap-3 ${
              !selectedCell || currentLevel >= MAX_LEVEL
                ? "bg-gray-400 text-gray-200 border-gray-300 cursor-pointer"
                : "bg-gray-900 text-white border-gray-800 font-semibold"
            }`}
          >
            <Command className="h-4 w-4 mr-2" />
            Analyze
          </button>
          <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
            Projection
          </button>
        </div>

        <div className="flex flex-wrap gap-4">
          <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
            Under the hood
          </button>
          <button
            onClick={handleWorkbench}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Go to work bench
            <ChevronDownIcon className="h-4 w-4 ml-2" />
          </button>
        </div>
      </div>
      {showModal && (
        <WorkbenchModal
          categorical_columns={categorical_columns1}
          currentLevel={currentLevel}
          currentPath={currentPath}
          activeKPI={newkpi}
          showModal={showModal}
          setShowModal={setShowModal}
          project_id={project_id}
        />
      )}
    </div>
  );
}
