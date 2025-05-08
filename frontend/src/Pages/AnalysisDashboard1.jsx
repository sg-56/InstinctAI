export default function AnalysisDashboard1() {
  return (
    <div className="min-h-[calc(100vh-65px)] bg-gray-50 p-6">
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 max-w-7xl mx-auto">
        {/* Left Card - 30% */}
        <div className="md:col-span-4">
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xl font-bold">Just</span>
              <div className="w-6 h-6 bg-red-500 rounded"></div>
              <span className="text-xl font-bold">Books</span>
            </div>
            <div className="space-y-2 text-sm text-gray-600">
              <p>Analyzing Revenue</p>
              <p>Optimizing revenue by addressing renewals</p>
              <p>Analyzing Parameter 3</p>
            </div>
          </div>
        </div>

        {/* Right Content - 70% */}
        <div className="md:col-span-8">
          <h2 className="text-lg font-semibold mb-6">Pointers on level 1</h2>

          {/* Parameters Table */}
          <div className="mb-8 overflow-x-auto">
            <table className="w-full border-collapse bg-white">
              <thead>
                <tr className="border-b">
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-900">Title</th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-900">Cluster 1</th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-900">Cluster 2</th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-900">Cluster 3</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 3 }).map((_, index) => (
                  <tr key={index} className="border-b">
                    <td className="py-3 px-4 text-sm text-gray-900">Parameter</td>
                    <td className="py-3 px-4 text-sm text-gray-900">Parameter 1</td>
                    <td className="py-3 px-4 text-sm text-gray-900">Parameter 2</td>
                    <td className="py-3 px-4 text-sm text-gray-900">Parameter 3</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Analyze Button */}
          <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-black rounded-md mb-6">
            <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
            Analyze
          </button>

          {/* Define Action Section */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Define action</h3>
            <textarea
              className="w-full h-32 p-3 border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your action here..."
            />
          </div>

          {/* Action Button */}
          <div className="flex justify-end mb-6">
            <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-black rounded-md">
              Action it
              <svg className="w-4 h-4 ml-2" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          {/* Bottom Navigation */}
          <div className="flex gap-4">
            <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
              <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                />
              </svg>
              Under the hood
            </button>
            <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
              <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Download
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

