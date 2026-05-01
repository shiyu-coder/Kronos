import { cn } from "@/lib/utils";

const WaveLoader = ({ className, primaryBgClass }: { className?: string; primaryBgClass?: string }) => {
  return (
    <div className={`flex gap-0.5 ${className}`}>
      {[...Array(10)].map((_, i) => (
        <div
          key={i}
          className={cn(`w-1 h-4 bg-primary animate-[pulse_1.5s_infinite]`, primaryBgClass)}
          style={{ animationDelay: `${i * 0.1}s` }}
        />
      ))}
    </div>
  );
};

export default WaveLoader;
