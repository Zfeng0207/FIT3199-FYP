import React, { useEffect, useState } from "react";

export default function AboutUs() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("/api/about_us")
      .then(res => res.json())
      .then(setData)
      .catch(err => console.error("API error", err));
  }, []);

  if (!data) return <div className="p-6 text-center">Loading…</div>;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">About Us</h1>
      <p className="mb-6">{data.mission}</p>
      <div className="grid gap-6 md:grid-cols-2">
        {data.team.map(member => (
          <div
            key={member.id}
            className="border rounded-xl p-4 shadow hover:shadow-lg transition"
          >
            <h2 className="text-xl font-semibold">{member.name}</h2>
            <p className="text-sm text-gray-600">{member.role}</p>
            <p className="mt-2">{member.bio}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
