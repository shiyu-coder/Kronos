"use client";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import Link from "next/link";

type SiteHeaderProps = {
  savedResultsCount: number;
};

export function SidebarHeader({ savedResultsCount }: SiteHeaderProps) {
  return (
    <header className='flex py-4 shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)'>
      <div className='flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6 '>
        <SidebarTrigger className='-ml-1' />
        <Separator orientation='vertical' className='data-[orientation=vertical]' />
        <p className='text-xs text-muted-foreground'>
          {savedResultsCount} saved {savedResultsCount === 1 ? "result" : "results"}
        </p>

        <div className='ml-auto flex items-center gap-2'>
          <Button asChild variant='outline' size='sm' className='ml-4'>
            <Link href='/'>
              <HugeiconsIcon icon={ArrowLeft01Icon} />
              Studio
            </Link>
          </Button>
        </div>
      </div>
    </header>
  );
}
