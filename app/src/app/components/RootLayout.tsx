import { Outlet } from "react-router";

export function RootLayout() {
  return (
    <div className="min-h-screen bg-gray-100 flex justify-center w-full font-sans text-gray-900">
      <div className="w-full max-w-md bg-white min-h-screen shadow-xl overflow-hidden relative flex flex-col">
        <Outlet />
      </div>
    </div>
  );
}
