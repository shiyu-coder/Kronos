import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

const LabelWithTooltip = ({ label, description }: { label: string; description: string }) => (
  <TooltipProvider>
    <div className='flex items-center gap-1'>
      <h2 className='text-sm font-semibold'>{label}</h2>
      <Tooltip>
        <TooltipTrigger asChild>
          <button className='rounded-full p-0.5 hover:bg-accent'>
            <HugeiconsIcon icon={InformationCircleIcon} className='size-3 text-muted-foreground' />
          </button>
        </TooltipTrigger>
        <TooltipContent className='bg-primary'>
          <p className='max-w-64 text-xs'>{description}</p>
        </TooltipContent>
      </Tooltip>
    </div>
  </TooltipProvider>
);

export default LabelWithTooltip;
